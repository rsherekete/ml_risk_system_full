"""End-to-end: run the dashboard's real pipeline against BigQuery, 30-day window.

30 days rather than 90 deliberately -- the memory guard exists because a
90-day multi-server BigQuery pull is ~15 GB, and 30 days of BigQuery data is
richer than 90 days of the pruned MySQL tables anyway.
"""
import sys, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from trading_data.behaviour_features import build_active_day_frame, feature_columns
from trading_data.bigquery_data_client import BQ_DATASET_FOR_DATABASE, BigQueryDataClient, BigQueryUnavailableError, is_demo_database
from trading_data.book_assignment import assign_books, daily_intelligence_components, expanding_account_pnl_volatility, expanding_client_profile, firm_daily_pnl
from trading_data.research import Economics, add_provenance

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start, end = decision_day - pd.Timedelta(days=30), decision_day + pd.Timedelta(days=1)
MIN_TRAIN, CADENCE = 20, 5

frames, provenance = [], []
t0 = time.time()
for database in BQ_DATASET_FOR_DATABASE:
    if is_demo_database(database):
        continue
    try:
        client = BigQueryDataClient(database)
        frame = add_provenance(client.get_trade_records(start.to_pydatetime(), end.to_pydatetime()), database)
        frames.append(frame)
        provenance.append({"database": database, "rows": len(frame), "accounts": frame["account_key"].nunique()})
        print(f"  {database:<18} {len(frame):>9,} rows  {frame['account_key'].nunique():>6,} accounts", flush=True)
    except BigQueryUnavailableError as exc:
        print(f"  {database:<18} skipped ({str(exc)[:60]}...)", flush=True)

records = pd.concat(frames, ignore_index=True)
del frames
gb = records.memory_usage(deep=True).sum() / 1e9
print(f"\nrecords: {len(records):,} rows, {records['account_key'].nunique():,} accounts, {gb:.2f} GB [{time.time()-t0:.0f}s]", flush=True)

print("\nbuilding rich features + walk-forward classifier...", flush=True)
t0 = time.time()
frame = build_active_day_frame(records, max_gap_days=None)
columns = feature_columns(frame)
for column in columns:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
frame["decision_day"] = pd.to_datetime(frame["decision_day"]).dt.normalize()
days = sorted(frame["decision_day"].unique())
print(f"  {len(frame):,} labelled rows, {len(columns)} features, {len(days)} days", flush=True)

import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)
pred = pd.Series(np.nan, index=frame.index, dtype="float64")
fitted = False
for offset in range(MIN_TRAIN, len(days)):
    test_mask = frame["decision_day"] == days[offset]
    if not test_mask.any():
        continue
    if (offset - MIN_TRAIN) % CADENCE == 0 or not fitted:
        train_mask = frame["decision_day"].isin(days[:offset])
        y = frame.loc[train_mask, "target_client_wins"]
        if y.notna().sum() > 100 and y.astype(bool).nunique() >= 2:
            model.fit(frame.loc[train_mask, columns], y.astype(bool))
            fitted = True
    if fitted:
        pred.loc[test_mask] = model.predict_proba(frame.loc[test_mask, columns])[:, 1]

from trading_data.research import _rank_discrimination
valid = pred.notna()
if valid.any():
    disc = _rank_discrimination(pred[valid].to_numpy(), frame.loc[valid, "target_client_wins"].astype(bool).to_numpy())
    print(f"  scored {int(valid.sum()):,} OOS rows, ROC AUC {disc['roc_auc']:.4f} [{time.time()-t0:.0f}s]", flush=True)

scored = frame.loc[valid, ["account_key", "database", "platform", "decision_day"]].copy()
scored["model_probability_loss"] = 1.0 - pred[valid].to_numpy()

print("\nrouting + firm P&L...", flush=True)
t0 = time.time()
components = daily_intelligence_components(records)
profile = expanding_client_profile(components)
volatility = expanding_account_pnl_volatility(components)
assignment = assign_books(scored, profile, volatility, 70, 30, economics=Economics())
daily_pnl, detail, metrics = firm_daily_pnl(records, assignment, Economics())
print(f"  assignment: {assignment['account_key'].nunique():,} accounts  |  "
      f"{assignment['book'].value_counts().to_dict()}", flush=True)
print(f"  source: {assignment['expected_value_source'].iloc[0]}", flush=True)
print(f"  detail covers {detail['account_key'].nunique():,} accounts (records has {records['account_key'].nunique():,})", flush=True)
print(f"  firm P&L: ${metrics['total_firm_pnl_usd']:,.0f}  max DD ${metrics['max_daily_drawdown_usd']:,.0f}  "
      f"win-day {metrics['win_day_rate']:.0%}  [{time.time()-t0:.0f}s]", flush=True)
print("\nEND-TO-END BIGQUERY PIPELINE OK", flush=True)
