"""How much did the open-time-ordered walk-forward actually leak?

Two distinct questions:
  1. At DECISION time we obviously do not wait for the trade to close -- the
     model scores at entry. That was never in dispute.
  2. At TRAINING time the label IS the realised profit, which does not exist
     until the trade closes. Training at day t on a position still running is
     using an outcome nobody had yet.

Whether (2) matters is empirical: if trades close within hours, almost nothing
leaks. This measures it, and also checks a second issue -- the extract selects
on CLOSE date, so early open-days may contain only unusually long-held trades.
"""
import pandas as pd
import numpy as np

BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
parts = []
for server in ("mt4_live01", "mt4_live02", "mt4_live04"):
    part = pd.read_parquet(f"{BASE}\\bq_90d_records.parquet",
                           columns=["state", "open_time", "close_time", "net_profit", "volume_lots"],
                           filters=[("database", "==", server)])
    parts.append(part.loc[(part["state"].astype("string") == "closed")
                          & part["open_time"].notna() & part["close_time"].notna()
                          & part["net_profit"].notna() & (part["volume_lots"] > 0)])
trades = pd.concat(parts, ignore_index=True)

hours = (trades["close_time"] - trades["open_time"]).dt.total_seconds() / 3600
print(f"{len(trades):,} closed trades")
print("\nholding time:")
for q in (0.5, 0.75, 0.9, 0.95, 0.99):
    print(f"  p{int(q*100):<3} {hours.quantile(q):>12,.1f} hours")
print(f"  mean {hours.mean():,.1f} h | max {hours.max():,.0f} h")
print(f"\n  closed within 24h : {(hours <= 24).mean():>6.1%}")
print(f"  open  >  7 days   : {(hours > 24*7).mean():>6.1%}")
print(f"  open  > 30 days   : {(hours > 24*30).mean():>6.1%}")

open_day = trades["open_time"].dt.floor("D")
close_day = trades["close_time"].dt.floor("D")
print(f"\ndistinct OPEN days  : {open_day.nunique()}")
print(f"distinct CLOSE days : {close_day.nunique()}")
print(f"open  range: {open_day.min().date()} .. {open_day.max().date()}")
print(f"close range: {close_day.min().date()} .. {close_day.max().date()}")

# Selection bias check: the extract filters on CLOSE date, so trades opened long
# before the window can only appear if they stayed open long enough to close
# inside it. Early open-days should therefore contain only long-held positions.
window_start = close_day.min()
early = open_day < window_start
print(f"\ntrades opened BEFORE the close-window starts: {early.mean():.1%} "
      f"({early.sum():,} rows)")
if early.any():
    print(f"  their median holding time: {hours[early].median():,.1f} h")
    print(f"  vs trades opened inside the window: {hours[~early].median():,.1f} h")
    print("  (a large gap means early open-days are a biased sample of survivors,")
    print("   which distorts any walk-forward that iterates over open dates)")
