import sys
sys.path.insert(0, r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad")
from google.cloud import bigquery
from markout_features import markout_sql

client = bigquery.Client(project="zfx-dwh-prod")

# What server_name values exist, so they can be mapped to our database names?
q = """
SELECT server_name, COUNT(*) AS n, MIN(DATE(close_time)) AS first_day, MAX(DATE(close_time)) AS last_day
FROM `zfx-dwh-prod.markouts_v2.markouts`
WHERE close_time_ms >= TIMESTAMP('2026-08-20') AND close_time_ms < TIMESTAMP('2026-08-28')
GROUP BY server_name ORDER BY n DESC
"""
job = client.query(q)
print("server_name values (last 8 days):")
for row in job.result():
    print(f"  {row['server_name']:<28} {row['n']:>10,}  {row['first_day']} -> {row['last_day']}")
print(f"  [scanned {job.total_bytes_processed/1e9:.2f} GB]\n")

# Dry-run the real 90-day aggregation to price it before running it.
sql = markout_sql("2026-05-29", "2026-08-28")
dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
gb = dry.total_bytes_processed / 1e9
print(f"90-day markout aggregation would scan {gb:.1f} GB (~${gb/1000*6.25:.2f})")
