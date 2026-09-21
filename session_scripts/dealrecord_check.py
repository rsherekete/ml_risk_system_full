from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

sql = """
    SELECT MAX(ts) as max_ts, MIN(ts) as min_ts, COUNT(*) as cnt
    FROM `zfx-dwh-prod.mt5_live01.dealrecord`
"""
job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
dry = client.query(sql, job_config=job_config)
est_mb = dry.total_bytes_processed / (1024**2)
print(f"DRY RUN estimated bytes: {dry.total_bytes_processed} ({est_mb:.2f} MB)")

if est_mb < 2000:
    job = client.query(sql)
    rows = list(job.result())
    print(f"ACTUAL bytes_processed: {job.total_bytes_processed} ({job.total_bytes_processed/(1024**2):.2f} MB)")
    for r in rows:
        print(f"  {dict(r)}")
else:
    print("SKIPPING - too expensive")
