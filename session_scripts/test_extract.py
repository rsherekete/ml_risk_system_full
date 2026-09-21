"""Verify the MySQL extractor on one month before committing to a 2-year run."""
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import mysql_extract

end = datetime(2026, 8, 1, tzinfo=timezone.utc)
start = datetime(2026, 7, 1, tzinfo=timezone.utc)

for database in ("mt4_live03", "mt5_live01"):
    t0 = time.time()
    try:
        frame = mysql_extract.fetch_month(database, start, end)
    except Exception as error:
        print(f"{database}: FAILED {type(error).__name__}: {str(error)[:200]}")
        continue
    print(f"\n{database}: {len(frame):,} trades in {time.time() - t0:.0f}s")
    if frame.empty:
        continue
    print(f"  window     : {frame['open_time'].min()} .. {frame['close_time'].max()}")
    print(f"  client P&L : ${frame['net_profit'].sum():,.0f}")
    print(f"  direction  : {frame['cmd'].value_counts().to_dict()}")
    print(f"  lots       : median {frame['volume_lots'].median():.2f}, max {frame['volume_lots'].max():.2f}")
    hold = (frame["close_time"] - frame["open_time"]).dt.total_seconds() / 3600
    print(f"  hold hours : median {hold.median():.2f}, negative {int((hold < 0).sum())}")
    # A buy closing above entry should be profitable; a low rate means the
    # direction was taken from the wrong side of the pair.
    buys = frame.loc[frame["cmd"] == "buy"]
    if len(buys):
        agree = ((buys["close_price"] > buys["open_price"]) == (buys["net_profit"] > 0)).mean()
        print(f"  direction sanity: {agree:.1%} of buys agree (price up <-> profit)")
    print(frame.head(2).to_string()[:700])

print("\n--- open positions ---")
for database in ("mt4_live03", "mt5_live01"):
    try:
        positions = mysql_extract.open_positions(database)
        lots = positions["volume_lots"].abs().sum() if len(positions) else 0
        print(f"{database}: {len(positions):,} open positions, {lots:,.2f} lots")
    except Exception as error:
        print(f"{database}: FAILED {type(error).__name__}: {str(error)[:120]}")
