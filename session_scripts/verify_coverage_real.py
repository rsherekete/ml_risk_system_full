import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import time
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.research import (
    account_group_lookup, add_provenance, clients_from_yaml, daily_account_features,
    fit_oos_predictions, non_client_logins, platform_for_database, predict_live, supervised_dataset,
)
from trading_data.book_assignment import (
    assign_books, daily_intelligence_components, expanding_client_profile,
    firm_daily_pnl, firm_daily_pnl_fixed_book,
)

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start = decision_day - pd.Timedelta(days=90)
end = decision_day + pd.Timedelta(days=1)

clients = clients_from_yaml()
name = "mt4_live01"
client = clients[name]
t0 = time.time()

frame = add_provenance(client.get_trade_records(start.to_pydatetime(), end.to_pydatetime()), name)
flagged = non_client_logins(account_group_lookup(client, platform_for_database(name), frame.get("login", pd.Series(dtype="float64"))))
if flagged:
    frame = frame.loc[~frame["login"].astype("Int64").isin(flagged)].copy()
print(f"{name}: {len(frame):,} rows, {frame['account_key'].nunique():,} distinct accounts (fetch: {time.time()-t0:.1f}s)")

records = frame
features = daily_account_features(records)
dataset = supervised_dataset(features)
print(f"features: {features['account_key'].nunique():,} accounts, dataset (labeled): {dataset['account_key'].nunique():,} accounts")

predictions = fit_oos_predictions(dataset, min_train_days=20)
live = predict_live(dataset, features, min_train_days=20)
predictions_with_live = pd.concat([predictions, live], ignore_index=True) if not live.empty else predictions
print(f"predictions: {predictions['account_key'].nunique():,} accounts, live: {live['account_key'].nunique():,} accounts, combined: {predictions_with_live['account_key'].nunique():,} accounts")

components = daily_intelligence_components(records)
profile = expanding_client_profile(components)
assignment = assign_books(predictions_with_live, profile, profit_weight=70, drawdown_weight=30)
print(f"\nassign_books book counts:\n{assignment['book'].value_counts()}")
print(f"assign_books distinct accounts: {assignment['account_key'].nunique():,}")

economics_default = __import__("trading_data").Economics()
daily_pnl, detail, metrics = firm_daily_pnl(records, assignment, economics_default)
print(f"\nfirm_daily_pnl detail distinct accounts: {detail['account_key'].nunique():,}")
print(f"records distinct accounts: {records['account_key'].nunique():,}")
print(f"COVERAGE CHECK: detail == records? {detail['account_key'].nunique() == records['account_key'].nunique()}")

today_row = daily_pnl.loc[daily_pnl['decision_day'] == daily_pnl['decision_day'].max()]
if not today_row.empty:
    print(f"\nmost recent day A-book accounts: {int(today_row['a_book_accounts'].iloc[0])}, B-book accounts: {int(today_row['b_book_accounts'].iloc[0])}, sum: {int(today_row['a_book_accounts'].iloc[0])+int(today_row['b_book_accounts'].iloc[0])}")

print(f"\nfirm_daily_pnl metrics: {metrics}")

always_a = firm_daily_pnl_fixed_book(records, "A_BOOK", economics_default)
always_b = firm_daily_pnl_fixed_book(records, "B_BOOK", economics_default)
print(f"\nAlways-A-book total pnl: ${always_a['firm_pnl_usd'].sum():,.0f}")
print(f"Always-B-book total pnl: ${always_b['firm_pnl_usd'].sum():,.0f}")
print(f"ML-routed total pnl:     ${metrics['total_firm_pnl_usd']:,.0f}")

print(f"\nTotal wall time: {time.time()-t0:.1f}s")
