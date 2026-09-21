import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
from google.cloud import bigquery
client = bigquery.Client(project="zfx-dwh-prod")

sql = """
SELECT `order`, login, NULLIF(NULLIF(symbol_name,'0'),'') AS symbol, symbol64 AS symbol_id,
  cmd AS cmd_raw, volume, open_ts, state AS state_raw,
  CAST(open_price AS FLOAT64) AS open_price, CAST(sl AS FLOAT64) AS sl, CAST(tp AS FLOAT64) AS tp,
  close_ts, reason AS reason_raw, CAST(close_price AS FLOAT64) AS close_price,
  CAST(profit AS FLOAT64) AS profit, CAST(commission AS FLOAT64) AS commission,
  CAST(storage AS FLOAT64) AS swap, CAST(taxes AS FLOAT64) AS taxes,
  gw_open_price, gw_close_price, CAST(margin_rate AS FLOAT64) AS margin_rate,
  TIMESTAMP_SECONDS(COALESCE(NULLIF(close_ts,0), open_ts)) AS timestamp
FROM `zfx-dwh-prod.mt4_demo_rep.orders`
WHERE COALESCE(NULLIF(close_ts,0), open_ts) >= UNIX_SECONDS(TIMESTAMP("2026-08-25"))
  AND COALESCE(NULLIF(close_ts,0), open_ts) <  UNIX_SECONDS(TIMESTAMP("2026-08-28"))
"""
job = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
b = job.total_bytes_processed
print(f"mt4_demo01 (mt4_demo_rep.orders) 3-day query: {b:,} bytes ({b/1e9:.2f} GB) ~= ${b/1e12*6.25:.4f} per refresh")
print("NOTE: table is UNPARTITIONED, so this cost is the same for a 3-day or 6-month window.")
