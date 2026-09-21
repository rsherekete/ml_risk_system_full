import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import time
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.research import (
    Economics, account_group_lookup, add_provenance, clients_from_yaml, daily_account_features,
    fit_oos_expected_value, non_client_logins, platform_for_database, predict_live_expected_value,
    supervised_dataset,
)
from trading_data.book_assignment import (
    assign_books, daily_intelligence_components, expanding_account_pnl_volatility,
    expanding_client_profile, firm_daily_pnl, real_efficient_frontier,
)

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start = decision_day - pd.Timedelta(days=90)
end = decision_day + pd.Timedelta(days=1)

t0 = time.time()
frames = []
for name, client in clients_from_yaml().items():
    frame = add_provenance(client.get_trade_records(start.to_pydatetime(), end.to_pydatetime()), name)
    flagged = non_client_logins(account_group_lookup(client, platform_for_database(name), frame.get("login", pd.Series(dtype="float64"))))
    if flagged:
        frame = frame.loc[~frame["login"].astype("Int64").isin(flagged)].copy()
    frames.append(frame)
    print(f"{name}: {len(frame):,} rows, {frame['account_key'].nunique():,} accounts")
records = pd.concat(frames, ignore_index=True)
print(f"\nTOTAL: {len(records):,} rows, {records['account_key'].nunique():,} distinct accounts ({time.time()-t0:.1f}s)")

features = daily_account_features(records)
dataset = supervised_dataset(features)
print(f"features: {features['account_key'].nunique():,} accounts, dataset (labeled): {dataset['account_key'].nunique():,} accounts")

expected_value = fit_oos_expected_value(dataset, min_train_days=20)
live_expected_value = predict_live_expected_value(dataset, features, min_train_days=20)
expected_value_with_live = pd.concat([expected_value, live_expected_value], ignore_index=True) if not live_expected_value.empty else expected_value
print(f"expected_value: {expected_value['account_key'].nunique():,} accounts, live: {live_expected_value['account_key'].nunique():,} accounts, combined: {expected_value_with_live['account_key'].nunique():,} accounts")

components = daily_intelligence_components(records)
profile = expanding_client_profile(components)
volatility = expanding_account_pnl_volatility(components)
print(f"volatility sources: {volatility['pnl_vol_source'].value_counts().to_dict()}")

t1 = time.time()
economics = Economics()
frontier = real_efficient_frontier(records, expected_value_with_live, profile, volatility, economics, steps=21)
print(f"\nreal_efficient_frontier: 21 points in {time.time()-t1:.1f}s\n")
print(frontier[["profit_weight", "total_firm_pnl_usd", "max_daily_drawdown_usd", "is_pareto_optimal"]].to_string(index=False))

pnl = frontier.sort_values("profit_weight")["total_firm_pnl_usd"].to_numpy()
deltas = pnl[1:] - pnl[:-1]
print(f"\nstep-to-step total_firm_pnl_usd deltas: {[round(d, 2) for d in deltas]}")
is_monotonic = bool((deltas >= -1e-6).all())
print(f"\ntotal_firm_pnl_usd monotonically non-decreasing across the sweep: {is_monotonic}")
if is_monotonic:
    print("CONFIRMED FIXED: the risk-budget-walk redesign is monotonic on the REAL data that originally exposed the bug.")
else:
    worst = deltas.min()
    print(f"STILL NOT MONOTONIC -- worst single-step decrease: {worst:,.2f}. This needs further investigation before claiming the fix works.")

# sanity: coverage invariant still holds under the new logic
assignment_check = assign_books(expected_value_with_live, profile, volatility, profit_weight=70, drawdown_weight=30, economics=economics)
_, detail_check, _ = firm_daily_pnl(records, assignment_check, economics)
print(f"\ncoverage check @ 70/30: assignment accounts={assignment_check['account_key'].nunique():,}, "
      f"detail accounts={detail_check['account_key'].nunique():,}, records accounts={records['account_key'].nunique():,}")
print(f"book counts @ 70/30:\n{assignment_check['book'].value_counts()}")

print(f"\nTotal wall time: {time.time()-t0:.1f}s")
