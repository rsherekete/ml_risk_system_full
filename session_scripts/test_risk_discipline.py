"""Does risk-management data finally make MAGNITUDE predictable?

Six attempts have failed to rank loss magnitude (all Spearman ~0), every one
of them using behavioural features that cannot see position size relative to
equity. This adds the denominators -- balance, margin, margin_level, free
margin -- and the sizing/stop discipline they enable.

The hypothesis is specific: |P&L| is bounded by how much of the account is at
risk, so knowing exposure-to-equity should make magnitude predictable where
behaviour alone could not.

Scoped to mt5_live01, the only server with full margin data (9,149 accounts).
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from google.cloud import bigquery
from trading_data.behaviour_features import build_active_day_frame, feature_columns
from trading_data.bigquery_data_client import compact_memory
from trading_data.research import _rank_discrimination
from trading_data.risk_discipline import RISK_FEATURE_COLUMNS, daily_risk_features, trade_risk_features

BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
START, END = "2026-05-29", "2026-08-28"
MIN_TRAIN, CADENCE = 20, 5
client = bigquery.Client(project="zfx-dwh-prod")

FINANCIALS_SQL = f"""
SELECT CAST(login AS INT64) AS login, DATE(ts) AS day,
       AVG(CAST(balance AS FLOAT64)) AS balance,
       AVG(CAST(margin AS FLOAT64)) AS margin,
       AVG(CAST(margin_free AS FLOAT64)) AS margin_free,
       AVG(CAST(margin_level AS FLOAT64)) AS margin_level,
       AVG(CAST(margin_leverage AS FLOAT64)) AS margin_leverage,
       AVG(CAST(profit AS FLOAT64)) AS floating_profit,
       SUM(CAST(daily_profit AS FLOAT64)) AS daily_profit,
       AVG(CAST(balance AS FLOAT64) + CAST(profit AS FLOAT64)) AS equity
FROM `zfx-dwh-prod.mt5_live01.dailyrecord`
WHERE ts >= TIMESTAMP('{START}') AND ts < TIMESTAMP('{END}')
GROUP BY login, day
"""

t0 = time.time()
financials = client.query(FINANCIALS_SQL).to_dataframe()
financials["account_key"] = "mt5_live01:" + financials["login"].astype("int64").astype(str)
financials["day"] = pd.to_datetime(financials["day"])
print(f"financials: {len(financials):,} account-days, {financials['account_key'].nunique():,} accounts [{time.time()-t0:.0f}s]")
# Split dormant from funded before quoting any statistic. The raw medians are
# $0 equity / 0% margin_level, which reads as "everyone is stopped out" but
# actually means most rows are empty accounts: MT5 writes margin_level = 0 when
# there are NO OPEN POSITIONS. Quoting the pooled median would have been
# actively misleading about the population we model.
funded = financials.loc[financials["equity"] > 10]
with_positions = financials.loc[financials["margin"].fillna(0) > 0]
print(f"  dormant/empty: {1 - len(funded)/len(financials):.1%} of rows have equity <= $10")
print(f"  funded rows:   {len(funded):,} | median equity ${funded['equity'].median():,.0f}")
print(f"  open positions:{len(with_positions):,} rows | median margin_level "
      f"{with_positions['margin_level'].median():,.0f}%")
if len(with_positions):
    genuine_stress = (with_positions["margin_level"] < 100).mean()
    print(f"  genuinely below margin call (positions open): {genuine_stress:.2%}\n", flush=True)

records = compact_memory(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", filters=[("database", "==", "mt5_live01")]))
frame = build_active_day_frame(records, max_gap_days=None)
frame["day"] = pd.to_datetime(frame["day"])
base_columns = feature_columns(frame)
print(f"behavioural: {len(frame):,} rows, {len(base_columns)} features", flush=True)

risk_daily = daily_risk_features(financials)
records["day"] = pd.to_datetime(records["timestamp"]).dt.floor("D")
trade_risk = trade_risk_features(records, risk_daily[["account_key", "day", "equity_usd"]])
del records; gc.collect()

# suffixes=("", "_risk") keeps the behavioural column names intact -- both
# frames carry `max_notional`/`total_notional`, and a default merge would
# rename BOTH sides, invalidating the already-computed base_columns list.
frame = frame.merge(
    risk_daily.drop(columns=[c for c in ("equity_usd",) if c in risk_daily]),
    on=["account_key", "day"], how="left", suffixes=("", "_risk"),
)
frame = frame.merge(
    trade_risk.drop(columns=["trades"], errors="ignore"),
    on=["account_key", "day"], how="left", suffixes=("", "_risk"),
)
missing_base = [c for c in base_columns if c not in frame.columns]
if missing_base:
    raise RuntimeError(f"merge renamed behavioural columns: {missing_base}")
risk_columns = [c for c in RISK_FEATURE_COLUMNS if c in frame.columns]
for column in risk_columns:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
coverage = frame[risk_columns].notna().any(axis=1).mean()
print(f"risk features: {len(risk_columns)} columns, {coverage:.1%} of rows have at least one")

# The raw table is dominated by dormant/empty accounts (median equity $0), so
# its summary statistics say nothing about the population being modelled.
# Coverage on ACTIVE account-days is what decides whether this test can work at
# all: a feature present on 5% of rows cannot move an aggregate metric.
print("per-feature coverage on ACTIVE account-days (the rows actually scored):")
for column in risk_columns:
    filled = frame[column].notna().mean()
    if filled > 0:
        median = frame[column].median()
        print(f"  {column:<32} {filled:>6.1%} present   median {median:>14,.4f}")
    else:
        print(f"  {column:<32}  EMPTY -- no usable values")
usable_risk = [c for c in risk_columns if frame[c].notna().mean() >= 0.01]
print(f"\n{len(usable_risk)}/{len(risk_columns)} risk features present on >=1% of active rows\n", flush=True)
risk_columns = usable_risk

for column in base_columns:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
frame["decision_day"] = pd.to_datetime(frame["decision_day"]).dt.normalize()
frame["label_wins"] = frame["target_client_wins"].astype(bool)
frame["label_abs"] = frame["target_profit"].abs()
frame["label_log_abs"] = np.log1p(frame["label_abs"])
frame["label_pnl"] = frame["target_profit"]
days = sorted(frame["decision_day"].unique())


def run(columns, label):
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
                reg.fit(frame.loc[train, columns], frame.loc[train, "label_log_abs"]); ok[1] = True
        if ok[0]:
            p.loc[test_mask] = clf.predict_proba(frame.loc[test_mask, columns])[:, 1]
        if ok[1]:
            m.loc[test_mask] = reg.predict(frame.loc[test_mask, columns])
    valid = p.notna() & m.notna()
    auc = _rank_discrimination(p[valid].to_numpy(), frame.loc[valid, "label_wins"].to_numpy())["roc_auc"]
    mag_sp = float(pd.Series(m[valid].to_numpy()).rank().corr(frame.loc[valid, "label_abs"].rank()))
    print(f"  {label:<34} direction AUC {auc:.4f} | MAGNITUDE Spearman {mag_sp:+.4f}  [{time.time()-t0:.0f}s]", flush=True)
    return auc, mag_sp, p, m, valid


print("=== THE TEST: does risk data make magnitude predictable? ===")
auc_b, mag_b, *_ = run(base_columns, "behavioural only")
auc_r, mag_r, p, m, valid = run(base_columns + risk_columns, "behavioural + risk discipline")
print(f"\n  DELTA  direction {auc_r - auc_b:+.4f}   MAGNITUDE {mag_r - mag_b:+.4f}")
print(f"  (magnitude Spearman was ~0.00 in six prior attempts -- anything above ~0.15 is a real change)\n")

if len(risk_columns) > 3:
    auc_only, mag_only, *_ = run(risk_columns, "risk discipline ALONE")
    print()

scored = frame.loc[valid].copy()
scored["p_win"] = p[valid]
scored["expected_abs"] = np.expm1(m[valid]).clip(lower=0)
scored["hedge_value"] = (2 * scored["p_win"] - 1) * scored["expected_abs"]
baseline = -scored["label_pnl"].sum()
print(f"flat B-book (mt5_live01 only): ${baseline:,.0f}")
for pct in (0.05, 0.10, 0.20):
    cutoff = scored["hedge_value"].quantile(1 - pct)
    hedged = scored["hedge_value"] >= cutoff
    pnl = -scored.loc[~hedged, "label_pnl"].sum()
    random_pnl = baseline * (1 - pct)
    print(f"  hedge top {int(pct*100):>2}%: ${pnl:>14,.0f}  vs flat {pnl-baseline:>+14,.0f}  "
          f"vs random-hedge {pnl-random_pnl:>+14,.0f}")
print("\n(vs random-hedge > 0 means the ranking beat hedging an arbitrary same-sized subset)")

importance_columns = base_columns + risk_columns
reg_final = lgb.LGBMRegressor(n_estimators=200, verbose=-1, random_state=0)
fit_mask = frame["label_log_abs"].notna()
reg_final.fit(frame.loc[fit_mask, importance_columns], frame.loc[fit_mask, "label_log_abs"])
importance = pd.Series(reg_final.feature_importances_, index=importance_columns).sort_values(ascending=False)
print("\ntop 20 features for predicting |P&L| (risk features marked *):")
for name, value in importance.head(20).items():
    marker = " *" if name in risk_columns else ""
    print(f"  {name:<34} {value:>6}{marker}")
