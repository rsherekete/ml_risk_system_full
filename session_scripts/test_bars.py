"""Validate minute-bar extraction speed before committing to a 12-day pull."""
import sys
import time
from datetime import datetime

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import tick_bars

SYMBOLS = ("XAUUSD", "XAUUSDe", "XAUUSDmin")
start = datetime(2026, 8, 25, 0, 0)
end = datetime(2026, 8, 25, 6, 0)

t0 = time.time()
bars = tick_bars.fetch_bars("mt4_live01", SYMBOLS, start, end, chunk_hours=3)
elapsed = time.time() - t0

print(f"{len(bars):,} bars in {elapsed:.0f}s for {len(SYMBOLS)} symbols over 6 hours")
if bars.empty:
    raise SystemExit("no bars returned")
print(f"projected for 12 days x 3 symbols: {elapsed * 48 / 60:.1f} min")
print()
print(bars.groupby("symbol").agg(
    bars=("minute", "size"), ticks=("ticks", "sum"),
    first=("minute", "min"), last=("minute", "max"),
    low=("low", "min"), high=("high", "max")).to_string())
print("\nsample:")
print(bars.head(4).to_string(index=False))

# Sanity: high must bound close, low must bound close.
bad = bars.loc[(bars["high"] < bars["close"]) | (bars["low"] > bars["close"])]
print(f"\nbars with inconsistent OHLC: {len(bad)}  {'OK' if bad.empty else 'PROBLEM'}")
