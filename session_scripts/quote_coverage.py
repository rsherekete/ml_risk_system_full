"""What price history do we actually hold, and is it dense enough for a path-aware backtest?

A stop can only be evaluated honestly if we know whether price TOUCHED the level
between entry and exit. That needs quotes covering the trade's life at
reasonable density -- sparse coverage would systematically under-detect stops
and flatter the policy, which is exactly the bias the previous attempt had.
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

STORE = Path(r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts\live_stream.duckdb")
connection = duckdb.connect(str(STORE), read_only=True)

print("quote coverage:")
rows = connection.execute("""
    SELECT canonical, COUNT(*) AS ticks,
           MIN(event_time) AS first_tick, MAX(event_time) AS last_tick,
           COUNT(DISTINCT CAST(event_time AS DATE)) AS days
    FROM quotes WHERE canonical IS NOT NULL AND mid > 0
    GROUP BY 1 ORDER BY ticks DESC
""").fetchall()
for symbol, ticks, first, last, days in rows:
    span_hours = (last - first).total_seconds() / 3600
    print(f"  {symbol:<12} {ticks:>9,} ticks | {days:>2}d | "
          f"{first:%m-%d %H:%M} .. {last:%m-%d %H:%M} | "
          f"{ticks / max(1, span_hours):>8,.0f} ticks/hour")

total, first, last = connection.execute(
    "SELECT COUNT(*), MIN(event_time), MAX(event_time) FROM quotes").fetchone()
print(f"\ntotal {total:,} quotes, {first} .. {last}")

# Gap analysis: a long gap means a stop inside it cannot be detected.
print("\nlargest gaps per symbol (a stop inside a gap is undetectable):")
gaps = connection.execute("""
    WITH ordered AS (
      SELECT canonical, event_time,
             LAG(event_time) OVER (PARTITION BY canonical ORDER BY event_time) AS previous
      FROM quotes WHERE canonical IS NOT NULL
    )
    SELECT canonical,
           MAX(EPOCH(event_time) - EPOCH(previous)) / 60 AS max_gap_minutes,
           MEDIAN(EPOCH(event_time) - EPOCH(previous)) AS median_gap_seconds
    FROM ordered WHERE previous IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 12
""").fetchall()
for symbol, max_gap, median_gap in gaps:
    print(f"  {symbol:<12} max gap {max_gap:>8,.0f} min | median {median_gap:>6.1f}s")

connection.close()
