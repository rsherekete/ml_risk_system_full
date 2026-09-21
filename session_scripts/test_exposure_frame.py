"""Exercise the exposure-day feature builder on whatever the store already holds.

This is the riskiest new code path -- it redefines what an active day is, and
every feature and target downstream depends on it. Better to break it here than
two hours into a retrain.
"""
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import data_store
from webapp.exposure_days import exposure_days, summarise

end = datetime.now(timezone.utc)
start = end - timedelta(days=120)
t0 = time.time()
trades = data_store.read_history(databases=("mt4_live03",), start=start, end=end)
print(f"loaded {len(trades):,} trades in {time.time() - t0:.0f}s")
if trades.empty:
    raise SystemExit("store empty for that window")

trades["account_key"] = (trades["database"].astype(str) + ":"
                         + trades["login"].astype("int64").astype(str))
print(f"accounts: {trades['account_key'].nunique():,}")
print(f"window: {trades['open_time'].min()} .. {trades['close_time'].max()}")

t0 = time.time()
calendar = exposure_days(trades)
print(f"\nexposure calendar built in {time.time() - t0:.0f}s")
print(f"  {len(calendar):,} account-days")

facts = summarise(calendar)
for key, value in facts.items():
    print(f"  {key:<22} {value:,}" if isinstance(value, int) else f"  {key:<22} {value}")

# The comparison that justifies the redefinition: how many exposure days does
# the old closing-day definition simply not see?
closing_only = trades.assign(day=pd.to_datetime(trades["close_time"]).dt.normalize()) \
                     .groupby(["account_key", "day"]).size()
print(f"\nclosing-day definition : {len(closing_only):,} account-days")
print(f"exposure definition    : {len(calendar):,} account-days")
print(f"  -> {len(calendar) - len(closing_only):+,} "
      f"({(len(calendar) / max(1, len(closing_only)) - 1):+.1%})")

print("\nP&L conservation check (realised P&L must not be invented or lost):")
print(f"  trades total    ${trades['net_profit'].sum():,.2f}")
print(f"  calendar total  ${calendar['realised_pnl'].sum():,.2f}")
difference = abs(trades["net_profit"].sum() - calendar["realised_pnl"].sum())
print(f"  difference      ${difference:,.2f}  {'OK' if difference < 1 else 'MISMATCH'}")

print("\nsample of an account that carried positions across days:")
carrying = calendar.loc[calendar["carried"] == 1]
if not carrying.empty:
    account = carrying["account_key"].iloc[0]
    sample = calendar.loc[calendar["account_key"] == account].head(10)
    print(sample.to_string(index=False))
