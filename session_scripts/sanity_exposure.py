"""Is 5,124,921 exposure days a plausible number?

Three independent checks:
  1. accounts x active days -- does the arithmetic land near it?
  2. per-server ratios -- mt4_live04 reported MORE trades but FEWER exposure
     days than mt4_live01, which needs explaining rather than assuming.
  3. trades per exposure day -- an implausible value either way flags a bug.
"""
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import data_store
from webapp.exposure_days import exposure_days

end = datetime.now(timezone.utc)
start = end - timedelta(days=730)

print(f"{'server':<14}{'trades':>14}{'accounts':>11}{'days':>7}"
      f"{'exp.days':>12}{'/acct':>8}{'trades/day':>12}")
totals = {"trades": 0, "exposure": 0, "accounts": 0}

for server in ("mt4_live01", "mt4_live02", "mt4_live03", "mt4_live04", "mt5_live01"):
    chunk = data_store.read_history(
        databases=(server,), start=start, end=end,
        columns=["database", "login", "open_time", "close_time", "net_profit"])
    if chunk.empty:
        print(f"{server:<14} (empty)")
        continue
    chunk = chunk.loc[chunk["open_time"].notna() & chunk["net_profit"].notna()]
    chunk["account_key"] = (chunk["database"].astype(str) + ":"
                            + chunk["login"].astype("int64").astype(str))
    accounts = chunk["account_key"].nunique()
    span_days = (chunk["close_time"].max() - chunk["open_time"].min()).days

    calendar = exposure_days(chunk[["account_key", "open_time", "close_time", "net_profit"]])
    exposure = len(calendar)
    per_account = exposure / max(1, accounts)
    print(f"{server:<14}{len(chunk):>14,}{accounts:>11,}{span_days:>7}"
          f"{exposure:>12,}{per_account:>8.0f}{len(chunk)/max(1,exposure):>12.1f}")
    totals["trades"] += len(chunk)
    totals["exposure"] += exposure
    totals["accounts"] += accounts
    del chunk, calendar

print(f"\n{'TOTAL':<14}{totals['trades']:>14,}{totals['accounts']:>11,}"
      f"{'':>7}{totals['exposure']:>12,}")
print(f"\nsanity:")
print(f"  exposure days per account overall: {totals['exposure']/max(1,totals['accounts']):.0f}")
print(f"  if every account were active every day of 730: "
      f"{totals['accounts']*730:,} -- so we are at "
      f"{totals['exposure']/max(1,totals['accounts']*730):.1%} of the maximum")
print(f"  trades per exposure day: {totals['trades']/max(1,totals['exposure']):.1f}")
