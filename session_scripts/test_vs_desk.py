"""Would our routing have beaten the desk's ACTUAL decisions, in realised dollars?

Every "firm P&L" figure so far has been a counterfactual against a simulated
policy. `slippage_monitoring.book` records what the desk really did, so this
compares three policies on the same account-days using the same realised P&L:

    B-book everything      firm P&L = -client_pnl                (the de facto
                                                                  policy: the
                                                                  desk B-books
                                                                  99.7% of flow)
    desk's actual routing  A-booked -> ~0, B-booked -> -client_pnl
    our model, top N%      A-book the N% we rank highest, B-book the rest

A-book P&L is treated as ~0 market P&L (markup revenue is a separate, additive
layer and does not change the ranking between policies). Positive numbers are
firm profit.
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
from trading_data.execution_quality import SERVER_TO_DATABASE

RECORDS = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"
MIN_TRAIN, CADENCE = 20, 5

parts = []
for database in sorted(pd.read_parquet(RECORDS, columns=["database"])["database"].unique()):
    part = compact_memory(pd.read_parquet(RECORDS, filters=[("database", "==", database)]))
    parts.append(build_active_day_frame(part, max_gap_days=None))
    del part; gc.collect()
frame = pd.concat(parts, ignore_index=True)
del parts; gc.collect()

columns = feature_columns(frame)
for column in columns:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
frame["decision_day"] = pd.to_datetime(frame["decision_day"]).dt.normalize()
frame["label_wins"] = frame["target_client_wins"].astype(bool)
frame["label_client_pnl"] = frame["target_profit"]
frame["label_abs"] = frame["target_profit"].abs()
days = sorted(frame["decision_day"].unique())

p_win = pd.Series(np.nan, index=frame.index, dtype="float64")
mag = pd.Series(np.nan, index=frame.index, dtype="float64")
clf = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)
reg = lgb.LGBMRegressor(n_estimators=200, verbose=-1, random_state=0)
ok = [False, False]
t0 = time.time()
for offset in range(MIN_TRAIN, len(days)):
    test_mask = frame["decision_day"] == days[offset]
    if not test_mask.any():
        continue
    if (offset - MIN_TRAIN) % CADENCE == 0 or not all(ok):
        train_mask = frame["decision_day"].isin(days[:offset])
        y = frame.loc[train_mask, "label_wins"]
        if y.notna().sum() > 100 and y.nunique() >= 2:
            clf.fit(frame.loc[train_mask, columns], y); ok[0] = True
        if train_mask.sum() > 100:
            reg.fit(frame.loc[train_mask, columns], np.log1p(frame.loc[train_mask, "label_abs"])); ok[1] = True
    if ok[0]:
        p_win.loc[test_mask] = clf.predict_proba(frame.loc[test_mask, columns])[:, 1]
    if ok[1]:
        mag.loc[test_mask] = reg.predict(frame.loc[test_mask, columns])

valid = p_win.notna() & mag.notna()
scored = frame.loc[valid, ["account_key", "database", "decision_day", "day", "label_client_pnl", "label_wins"]].copy()
scored["p_win"] = p_win[valid]
scored["expected_abs"] = np.expm1(mag[valid]).clip(lower=0)
# Expected net benefit of hedging: E[client_pnl] = (2p-1) * E[|pnl|].
scored["expected_net_benefit"] = (2 * scored["p_win"] - 1) * scored["expected_abs"]
print(f"scored {len(scored):,} account-days [{time.time()-t0:.0f}s]", flush=True)

client = bigquery.Client(project="zfx-dwh-prod")
desk = client.query("""
SELECT server_name, account_number AS login, trading_date AS day,
       MAX(CASE WHEN UPPER(book) LIKE 'A%' THEN 1 ELSE 0 END) AS desk_a_book
FROM `zfx-dwh-prod.data_marts_multi_region.slippage_monitoring`
WHERE trading_date >= DATE('2026-05-29') AND trading_date < DATE('2026-08-28')
GROUP BY server_name, login, day
""").to_dataframe()
desk["database"] = desk["server_name"].map(SERVER_TO_DATABASE)
desk = desk.dropna(subset=["database", "login"])
desk["account_key"] = desk["database"] + ":" + desk["login"].astype("int64").astype(str)
desk["day"] = pd.to_datetime(desk["day"])
merged = scored.merge(desk[["account_key", "day", "desk_a_book"]], on=["account_key", "day"], how="inner")
print(f"matched {len(merged):,} account-days against the desk's actual routing "
      f"({merged['account_key'].nunique():,} accounts)\n", flush=True)

total_client_pnl = merged["label_client_pnl"].sum()
bbook_all = -total_client_pnl
desk_pnl = -merged.loc[merged["desk_a_book"] == 0, "label_client_pnl"].sum()
print(f"{'policy':<34}{'firm P&L':>16}{'vs B-book-all':>16}")
print(f"  {'B-book everything':<32}{bbook_all:>16,.0f}{'--':>16}")
print(f"  {'desk actual routing':<32}{desk_pnl:>16,.0f}{desk_pnl - bbook_all:>16,+.0f}")
print(f"    (desk A-booked {merged['desk_a_book'].mean():.2%} of account-days)")

print()
for pct in (0.01, 0.05, 0.10, 0.20, 0.30):
    cutoff = merged["expected_net_benefit"].quantile(1 - pct)
    hedged = merged["expected_net_benefit"] >= cutoff
    model_pnl = -merged.loc[~hedged, "label_client_pnl"].sum()
    print(f"  {'our model, hedge top ' + f'{int(pct*100)}%':<32}{model_pnl:>16,.0f}{model_pnl - bbook_all:>16,+.0f}")

print("\nPositive 'vs B-book-all' = selective hedging beat B-booking everything.")
print("Negative = the firm would have been better off just B-booking the lot.")
