"""Ingest all six servers plus quotes, and verify the derived risk queries."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import kafka_service as ks

store = Path(r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts\live_stream.duckdb")
if store.exists():
    store.unlink()

materialiser = ks.KafkaMaterialiser()
print(f"trade topics: {len(materialiser.topics)} | quote topics: {len(ks.QUOTE_TOPICS)}")
materialiser.start(backfill=True, with_quotes=True)

for step in range(16):
    time.sleep(5)
    coverage = materialiser.coverage()
    print(f"  t+{(step + 1) * 5:3}s  events={coverage.get('events', 0):>8,}  "
          f"span={coverage.get('span_days', 0):>6}d  symbols={coverage.get('symbols', 0):>4}",
          flush=True)
materialiser.stop()
time.sleep(2)

print("\nper-topic:")
for state in materialiser.status()["topics"]:
    print(f"  {state['topic'][-44:]:<46} {state['status']:<10} {state['consumed']:>9,}")

print("\nsymbol VaR (top 5):")
for row in materialiser.symbol_var()[:5]:
    print(f"  {row['symbol']:<12} days={row['days']:>3}  firm=${row['firm_pnl']:>12,.0f}  "
          f"1d=${row['var_1d']:>10,.0f}  5d=${row['var_5d']:>10,.0f}  20d=${row['var_20d']:>10,.0f}")

print(f"\nquote symbols: {materialiser.quote_symbols()[:12]}")
symbols = materialiser.quote_symbols()
if symbols:
    bars = materialiser.ohlc(symbols[0], minutes=5, hours_back=72)
    print(f"OHLC {symbols[0]}: {len(bars)} bars")
    for bar in bars[:3]:
        print("  ", bar)

print(f"\nopen positions (stream-derived): {len(materialiser.open_positions())}")
for row in materialiser.open_positions()[:5]:
    print("  ", row)
