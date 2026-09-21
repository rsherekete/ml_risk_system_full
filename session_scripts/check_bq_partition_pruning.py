import sys
from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

def dry_run_bytes(sql, params=None):
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False, query_parameters=params or [])
    job = client.query(sql, job_config=job_config)
    return job.total_bytes_processed

print("=== MT4 traderecord (view) -- 3-day window, filtered on tm ===")
sql_mt4_narrow = """
SELECT `order`, login, cmd, state, open_ts, close_ts, profit, commission, storage, taxes, tm
FROM `zfx-dwh-prod.mt4_live01.traderecord`
WHERE tm >= TIMESTAMP("2026-08-25") AND tm < TIMESTAMP("2026-08-28")
"""
b = dry_run_bytes(sql_mt4_narrow)
print(f"3-day window: {b:,} bytes ({b/1e9:.3f} GB)")

print("\n=== MT4 traderecord (view) -- 6-month window, filtered on tm ===")
sql_mt4_wide = """
SELECT `order`, login, cmd, state, open_ts, close_ts, profit, commission, storage, taxes, tm
FROM `zfx-dwh-prod.mt4_live01.traderecord`
WHERE tm >= TIMESTAMP("2026-02-28") AND tm < TIMESTAMP("2026-08-28")
"""
b2 = dry_run_bytes(sql_mt4_wide)
print(f"6-month window: {b2:,} bytes ({b2/1e9:.3f} GB)")
print(f"ratio wide/narrow: {b2/b:.1f}x (expect ~60x for 180 days vs 3 days if partition pruning works; near-1x means NO pruning through the view)")

print("\n=== MT5 deals (real table) -- 3-day window, filtered on time ===")
sql_mt5_narrow = """
SELECT deal, login, action, entry, price, profit, commission, storage, time, gateway, price_gateway
FROM `zfx-dwh-prod.mt5_live01.deals`
WHERE time >= TIMESTAMP("2026-08-25") AND time < TIMESTAMP("2026-08-28")
"""
b3 = dry_run_bytes(sql_mt5_narrow)
print(f"3-day window: {b3:,} bytes ({b3/1e9:.3f} GB)")

print("\n=== MT5 deals -- 6-month window, filtered on time ===")
sql_mt5_wide = """
SELECT deal, login, action, entry, price, profit, commission, storage, time, gateway, price_gateway
FROM `zfx-dwh-prod.mt5_live01.deals`
WHERE time >= TIMESTAMP("2026-02-28") AND time < TIMESTAMP("2026-08-28")
"""
b4 = dry_run_bytes(sql_mt5_wide)
print(f"6-month window: {b4:,} bytes ({b4/1e9:.3f} GB)")
print(f"ratio wide/narrow: {b4/b3:.1f}x")

print("\n=== MT4 underlying operational_data_store table directly (bypass view), 3-day, filtered on tm ===")
try:
    sql_mt4_direct = """
    SELECT `order`, login, cmd, state, open_ts, close_ts, profit, commission, storage, taxes, tm
    FROM `zfx-dwh-prod.operational_data_store.mt4_live01__traderecord`
    WHERE tm >= TIMESTAMP("2026-08-25") AND tm < TIMESTAMP("2026-08-28")
    """
    b5 = dry_run_bytes(sql_mt4_direct)
    print(f"3-day window (direct table): {b5:,} bytes ({b5/1e9:.3f} GB) -- compare to view's {b:,}")
except Exception as exc:
    print(f"FAILED -> {type(exc).__name__}: {exc}")
