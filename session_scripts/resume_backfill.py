"""Finish the 2-year backfill for the servers the VPN drop cut short.

Only months already on disk are skipped, so this resumes rather than restarting.
"""
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import data_store, mysql_extract

REMAINING = ("mt4_live02", "mt5_live01", "mt4_live01")
DAYS = 730

started = time.time()
for database in REMAINING:
    directory = data_store.WAREHOUSE / database
    have = {p.stem for p in directory.glob("*.parquet")} if directory.exists() else set()
    print(f"\n=== {database} ({len(have)} months already stored) ===", flush=True)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)
    written, failures = 0, []
    for month_start, month_end in mysql_extract.month_ranges(start, end):
        label = month_start.strftime("%Y-%m")
        if label in have:
            continue
        t0 = time.time()
        frame = None
        for attempt in range(4):
            try:
                frame = mysql_extract.fetch_month(database, month_start, month_end)
                break
            except Exception as error:
                if attempt == 3:
                    failures.append(f"{label}: {type(error).__name__}")
                    print(f"  {label}: FAILED after 4 attempts", flush=True)
                else:
                    wait = 20 * (attempt + 1)
                    print(f"  {label}: {type(error).__name__}, retry in {wait}s", flush=True)
                    time.sleep(wait)
        if frame is None:
            continue
        if not frame.empty:
            result = data_store.write_partitions(database, frame, time_column="close_time")
            written += result["written"]
        print(f"  {label}: {len(frame):,} rows in {time.time() - t0:.0f}s", flush=True)

    if written:
        data_store.set_watermark(database, datetime.now(timezone.utc), written, "mysql backfill")
    print(f"  -> {written:,} rows, {len(failures)} failed months", flush=True)

print(f"\ntotal {(time.time() - started) / 60:.1f} min")
summary = data_store.store_summary()
print(f"store: {summary['total_mb']:,.0f} MB across {summary['files']} files")
for name, info in summary["databases"].items():
    print(f"  {name:<18} {info['months']:>3} months  {info['first_month']} .. "
          f"{info['last_month']}  {info['size_mb']:>9,.1f} MB")
