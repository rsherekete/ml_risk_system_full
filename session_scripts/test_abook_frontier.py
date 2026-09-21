"""A-book selection + profit/drawdown Pareto frontier, on real 90-day data.

Also fixes the dtype bug that silently dropped most markout features: BigQuery
NUMERIC aggregates arrive as Python Decimal, which pandas types as `object`, so
an is_numeric_dtype filter discards them. Coerced explicitly here.
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from trading_data.abook_selection import build_abook_candidates, hedge_frontier
from trading_data.behaviour_features import build_active_day_frame, feature_columns
from trading_data.bigquery_data_client import compact_memory
from trading_data.research import _rank_discrimination

BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
MIN_TRAIN, CADENCE = 20, 5

parts = []
for database in sorted(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", columns=["database"])["database"].unique()):
    part = compact_memory(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", filters=[("database", "==", database)]))
    parts.append(build_active_day_frame(part, max_gap_days=None))
    del part; gc.collect()
frame = pd.concat(parts, ignore_index=True)
del parts; gc.collect()
base_columns = feature_columns(frame)

markouts = pd.read_parquet(f"{BASE}\\markout_all_servers.parquet")
# BigQuery NUMERIC -> Decimal -> pandas `object`. Coerce, or the horizon
# columns get silently dropped by any is_numeric_dtype filter downstream.
markout_columns = []
for column in markouts.columns:
    if column in {"database", "login", "day", "account_key"}:
        continue
    markouts[column] = pd.to_numeric(markouts[column], errors="coerce").astype("float32")
    markout_columns.append(column)
frame["day"] = pd.to_datetime(frame["day"])
frame = frame.merge(markouts[["account_key", "day", *markout_columns]], on=["account_key", "day"], how="left")
print(f"{len(frame):,} rows | behavioural {len(base_columns)} + markout {len(markout_columns)} "
      f"| coverage {frame['context_trades'].notna().mean():.1%}", flush=True)
print(f"markout features: {markout_columns}\n", flush=True)

frame["label_wins"] = frame["target_client_wins"].astype(bool)
frame["label_client_pnl"] = frame["target_profit"]
frame["label_abs"] = frame["target_profit"].abs()
frame["decision_day"] = pd.to_datetime(frame["decision_day"]).dt.normalize()
days = sorted(frame["decision_day"].unique())


def fit(columns, label):
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    p = pd.Series(np.nan, index=frame.index, dtype="float64")
    m = pd.Series(np.nan, index=frame.index, dtype="float64")
    clf = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)
    reg = lgb.LGBMRegressor(n_estimators=200, verbose=-1, random_state=0)
    ok = [False, False]
    t0 = time.time()
    for offset in range(MIN_TRAIN, len(days)):
        test_mask = frame["decision_day"] == days[offset]
        if not test_mask.any():
            continue
        if (offset - MIN_TRAIN) % CADENCE == 0 or not all(ok):
            train = frame["decision_day"].isin(days[:offset])
            y = frame.loc[train, "label_wins"]
            if y.notna().sum() > 100 and y.nunique() >= 2:
                clf.fit(frame.loc[train, columns], y); ok[0] = True
            if train.sum() > 100:
                reg.fit(frame.loc[train, columns], np.log1p(frame.loc[train, "label_abs"])); ok[1] = True
        if ok[0]:
            p.loc[test_mask] = clf.predict_proba(frame.loc[test_mask, columns])[:, 1]
        if ok[1]:
            m.loc[test_mask] = reg.predict(frame.loc[test_mask, columns])
    valid = p.notna() & m.notna()
    auc = _rank_discrimination(p[valid].to_numpy(), frame.loc[valid, "label_wins"].to_numpy())["roc_auc"]
    print(f"  {label:<26} ROC AUC {auc:.4f}  n={int(valid.sum()):,}  [{time.time()-t0:.0f}s]", flush=True)
    return p, m, valid, auc


print("=== does market context help? ===", flush=True)
_, _, _, auc_base = fit(base_columns, "behavioural only")
p, m, valid, auc_full = fit(base_columns + markout_columns, "behavioural + markouts")
print(f"  DELTA {auc_full - auc_base:+.4f}\n", flush=True)

wanted = [
    "account_key", "database", "decision_day", "label_client_pnl", "label_wins",
    "expanding_edge_flag", "expanding_arbitrage_flag", "life_win_rate", "life_profit_factor",
    "life_closes", "form_vs_life", "roll5_realised_pnl", "scalp_rate",
    "anticipation_5m", "markout_5m", "markout_positive_share",
]
# The edge/arbitrage flags come from `expanding_client_profile`, not from the
# behavioural feature builder, so take whatever is actually present rather than
# failing -- `classify_abook_candidate` already reads every optional column
# through .get() with a default.
available = [c for c in wanted if c in frame.columns]
missing = [c for c in wanted if c not in frame.columns]
if missing:
    print(f"  (not available for classification: {missing})", flush=True)
scored = frame.loc[valid, available].copy()
scored["probability_win"] = p[valid]
scored["expected_abs_pnl_usd"] = np.expm1(m[valid]).clip(lower=0)

candidates = build_abook_candidates(scored)
print("=== A-book candidate classification (whole window) ===", flush=True)
worth = candidates.loc[candidates["worth_hedging"]]
print(f"{len(worth):,} of {len(candidates):,} account-days have positive expected hedge value\n")
summary = worth.groupby("abook_category").agg(
    account_days=("account_key", "size"),
    accounts=("account_key", "nunique"),
    mean_hedge_value=("hedge_value_usd", "mean"),
    realised_client_pnl=("label_client_pnl", "sum"),
).sort_values("realised_client_pnl", ascending=False)
print(summary.to_string())
print("\n(realised_client_pnl > 0 means hedging that category actually SAVED money)")

print("\n=== profit / drawdown frontier ===", flush=True)
frontier = hedge_frontier(candidates, steps=21)
show = frontier.loc[frontier["hedge_fraction"].isin(frontier["hedge_fraction"].round(2).unique()[::2])]
print(show[["hedge_fraction", "accounts_hedged", "total_firm_pnl_usd", "max_drawdown_usd",
            "sharpe_like", "win_day_rate", "is_pareto_optimal"]].to_string(index=False))
best = frontier.loc[frontier["is_pareto_optimal"]]
print(f"\n{len(best)} Pareto-optimal settings out of {len(frontier)}")
if not best.empty:
    top = best.loc[best["total_firm_pnl_usd"].idxmax()]
    print(f"highest-profit Pareto point: hedge {top['hedge_fraction']:.0%} "
          f"-> P&L ${top['total_firm_pnl_usd']:,.0f}, max DD ${top['max_drawdown_usd']:,.0f}")
frontier.to_parquet(f"{BASE}\\hedge_frontier.parquet", index=False)
candidates.to_parquet(f"{BASE}\\abook_candidates.parquet", index=False)
