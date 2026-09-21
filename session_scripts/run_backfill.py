"""Two-year backfill from MySQL into the local partitioned store.

Runs smallest server first so problems surface in seconds rather than after an
hour of work on the largest table.
"""
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import data_store, mysql_extract

# Ascending by table size: mt4_live03 is 2.8M rows, mt4_live01 is 147.8M.
ORDER = ("mt4_live03", "mt4_live04", "mt4_live02", "mt5_live01", "mt4_live01")
DAYS = 730

started = time.time()
for database in ORDER:
    t0 = time.time()
    print(f"\n=== {database} ===", flush=True)
    result = mysql_extract.backfill(
        database, days=DAYS,
        progress=lambda message: print(f"  {message}", flush=True))
    print(f"  -> {result['rows']:,} rows across {len(result['months'])} months "
          f"in {time.time() - t0:.0f}s", flush=True)
    for failure in result["failures"]:
        print(f"  FAILED {failure}", flush=True)

print(f"\ntotal elapsed {(time.time() - started) / 60:.1f} min")
summary = data_store.store_summary()
print(f"store: {summary['total_mb']:,.0f} MB across {summary['files']} files")
for name, info in summary["databases"].items():
    print(f"  {name:<18} {info['months']:>3} months  {info['first_month']} .. "
          f"{info['last_month']}  {info['size_mb']:>8,.1f} MB")
