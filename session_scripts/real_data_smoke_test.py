import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.research import (
    Economics, add_provenance, attach_symbol_specs, clients_from_yaml,
    daily_account_features, supervised_dataset, fit_oos_predictions, predict_live,
    client_intelligence, symbol_mapping_table,
)
from trading_data.risk import daily_account_exposure, daily_symbol_drivers, symbol_var, portfolio_var
from trading_data.book_assignment import assign_books, daily_intelligence_components, expanding_client_profile, firm_daily_pnl, real_efficient_frontier
from trading_data.research import classifier_performance

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start = decision_day - pd.Timedelta(days=90)  # matches dashboard.py's default lookback
end = decision_day + pd.Timedelta(days=1)

frames, specs_by_db, failures = [], {}, []
for name, client in clients_from_yaml().items():
    try:
        frames.append(add_provenance(client.get_trade_records(start.to_pydatetime(), end.to_pydatetime()), name))
        specs_by_db[name] = client.symbol_specs()
    except Exception as exc:
        failures.append((name, exc))
        print(f"{name}: LOAD FAILED -> {type(exc).__name__}: {exc}")

records = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
print(f"records: {len(records):,} rows from {len(frames)} database(s)")
if records.empty:
    raise SystemExit("no records returned in the 14-day window -- nothing further to test")

enriched_frames = []
for name, specs in specs_by_db.items():
    subset = records.loc[records["database"] == name]
    if not subset.empty:
        enriched_frames.append(attach_symbol_specs(subset, specs))
records = pd.concat(enriched_frames, ignore_index=True)
print("notional_status counts:")
print(records["notional_status"].value_counts())
print("columns present -- commission:", "commission" in records.columns, "swap:", "swap" in records.columns, "net_profit:", "net_profit" in records.columns)

mapping = symbol_mapping_table(records, specs_by_db)
n_canonical = mapping["canonical_symbol"].nunique()
print(f"symbol_mapping_table: {len(mapping)} raw symbols -> {n_canonical} canonical")

features = daily_account_features(records)
dataset = supervised_dataset(features)
print(f"features: {len(features)} rows, dataset: {len(dataset)} rows")

predictions = fit_oos_predictions(dataset)  # dashboard.py default: min_train_days=20
live = predict_live(dataset, features)  # dashboard.py default: min_train_days=20
combined = pd.concat([predictions, live], ignore_index=True) if not live.empty else predictions
print(f"predictions: {len(predictions)}, live: {len(live)}")

components = daily_intelligence_components(records)
profile = expanding_client_profile(components)
assignment = assign_books(combined, profile, 70, 30)
daily_pnl, detail, metrics = firm_daily_pnl(records, assignment, Economics())
print("firm_daily_pnl metrics:", metrics)
print(f"detail covers {detail['account_key'].nunique()} distinct accounts overall (records has {records['account_key'].nunique()} total)")
latest_day = detail["decision_day"].max()
today_detail = detail.loc[detail["decision_day"] == latest_day]
print(f"latest decision_day in detail: {latest_day}, accounts on that day: {today_detail['account_key'].nunique()} "
      f"({(today_detail['book'] == 'A_BOOK').sum()} A_BOOK, {(today_detail['book'] == 'B_BOOK').sum()} B_BOOK)")

intel = client_intelligence(records)
print(f"client_intelligence: {len(intel)} accounts screened")

import time
t0 = time.time()
frontier = real_efficient_frontier(records, combined, profile, Economics(), steps=21)
print(f"real_efficient_frontier: {len(frontier)} points in {time.time()-t0:.1f}s, pareto-optimal: {frontier['is_pareto_optimal'].sum()}")
print(frontier[["profit_weight", "total_firm_pnl_usd", "max_daily_drawdown_usd", "is_pareto_optimal"]].to_string())

clf = classifier_performance(predictions, 70, 30)
print(f"classifier_performance @ 70/30: {clf}")

as_of = records["timestamp"].max()
if pd.notna(as_of):
    as_of = pd.Timestamp(as_of).floor("D")
    var_table = symbol_var(records, as_of)
    pf_var = portfolio_var(records, as_of)
    print(f"symbol_var rows: {len(var_table)}, portfolio_var rows: {len(pf_var)}")
    exposure = daily_account_exposure(records, as_of)
    drivers = daily_symbol_drivers(records, as_of, Economics())
    print(f"daily_account_exposure rows: {len(exposure)}, daily_symbol_drivers rows: {len(drivers)}")

print("REAL-DATA PIPELINE SMOKE TEST PASSED")
