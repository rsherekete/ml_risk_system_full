import sys
from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

# One day, mt4_live01: how much of the row volume is CDC churn vs distinct MySQL-equivalent rows?
sql = """
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT FORMAT('%d|%d', ts, sequence)) AS distinct_ts_sequence,
  COUNT(DISTINCT `order`) AS distinct_orders,
  COUNT(DISTINCT _cdc_op) AS distinct_cdc_ops,
  STRING_AGG(DISTINCT _cdc_op) AS cdc_ops,
  COUNTIF(state IN (3,4,5)) AS closed_state_rows,
  COUNT(DISTINCT CASE WHEN state IN (3,4,5) THEN FORMAT('%d|%d', ts, sequence) END) AS distinct_closed
FROM `zfx-dwh-prod.mt4_live01.traderecord`
WHERE tm >= TIMESTAMP("2026-08-26") AND tm < TIMESTAMP("2026-08-27")
"""
job = client.query(sql)
row = list(job.result())[0]
print(f"bytes processed: {job.total_bytes_processed:,}")
for k in row.keys():
    print(f"  {k}: {row[k]}")
