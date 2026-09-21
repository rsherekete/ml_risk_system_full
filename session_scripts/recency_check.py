from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

def run(label, sql, dry_run_first=True):
    if dry_run_first:
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        dry = client.query(sql, job_config=job_config)
        est_mb = dry.total_bytes_processed / (1024**2)
        print(f"[{label}] DRY RUN estimated bytes: {dry.total_bytes_processed} ({est_mb:.2f} MB)")
        if est_mb > 2000:  # >2GB, ask before running for real (print and skip actual run)
            print(f"[{label}] SKIPPING actual run - estimate too large ({est_mb:.2f} MB)")
            return
    job = client.query(sql)
    rows = list(job.result())
    print(f"[{label}] ACTUAL bytes_processed: {job.total_bytes_processed} ({job.total_bytes_processed/(1024**2):.2f} MB)")
    for r in rows:
        print(f"  {dict(r)}")

# deals: partitioned by `time` DAY, clustered position_id/entry
run("deals last 2 days", """
    SELECT COUNT(*) as cnt, MIN(time) as mn, MAX(time) as mx
    FROM `zfx-dwh-prod.mt5_live01.deals`
    WHERE time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 DAY)
""")

# orderrecord: partitioned by ts DAY, clustered login
run("orderrecord last 2 days", """
    SELECT COUNT(*) as cnt, MIN(ts) as mn, MAX(ts) as mx
    FROM `zfx-dwh-prod.mt5_live01.orderrecord`
    WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 DAY)
""")

# userrecord: partitioned ts DAY, clustered login
run("userrecord last 2 days", """
    SELECT COUNT(*) as cnt, MIN(ts) as mn, MAX(ts) as mx
    FROM `zfx-dwh-prod.mt5_live01.userrecord`
    WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 DAY)
""")

# positionrecord: partitioned ts DAY, clustered login
run("positionrecord last 2 days", """
    SELECT COUNT(*) as cnt, MIN(ts) as mn, MAX(ts) as mx
    FROM `zfx-dwh-prod.mt5_live01.positionrecord`
    WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 DAY)
""")

# ticks: partitioned ts DAY, clustered symbol -- huge table, check just 1 day
run("ticks last 1 day", """
    SELECT COUNT(*) as cnt, MIN(ts) as mn, MAX(ts) as mx, COUNT(DISTINCT symbol) as symbols, COUNT(DISTINCT bank) as banks, COUNT(DISTINCT feeder) as feeders
    FROM `zfx-dwh-prod.mt5_live01.ticks`
    WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
""")

# accounts (small, unpartitioned) -- just get max/min last_access as freshness signal, whole table scan is fine (30MB)
run("accounts freshness", """
    SELECT COUNT(*) as cnt, MAX(last_access) as max_last_access, MIN(registration) as min_reg
    FROM `zfx-dwh-prod.mt5_live01.accounts`
""", dry_run_first=True)
