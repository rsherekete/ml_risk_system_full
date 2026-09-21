from google.cloud import bigquery
client = bigquery.Client(project="zfx-dwh-prod")

def run(label, sql):
    job = client.query(sql)
    r = list(job.result())[0]
    print(f"{label}: {dict(r)}  [{job.total_bytes_processed/1e9:.2f} GB]")

# traderecord: every CDC event
run("traderecord 2d", """
SELECT COUNT(*) AS n_rows, COUNT(DISTINCT `order`) AS orders, COUNT(DISTINCT login) AS logins
FROM `zfx-dwh-prod.mt4_live01.traderecord`
WHERE tm >= TIMESTAMP("2026-08-25") AND tm < TIMESTAMP("2026-08-27")
""")

# orders: one row per order, partitioned on _close_ts_partition
run("orders closed 2d", """
SELECT COUNT(*) AS n_rows, COUNT(DISTINCT `order`) AS orders, COUNT(DISTINCT login) AS logins
FROM `zfx-dwh-prod.mt4_live01.orders`
WHERE _close_ts_partition >= TIMESTAMP("2026-08-25") AND _close_ts_partition < TIMESTAMP("2026-08-27")
""")

# traderecord restricted to genuinely-closed trade rows only (what the dashboard actually consumes)
run("traderecord 2d closed+trade-cmd", """
SELECT COUNT(*) AS n_rows, COUNT(DISTINCT `order`) AS orders, COUNT(DISTINCT login) AS logins
FROM `zfx-dwh-prod.mt4_live01.traderecord`
WHERE tm >= TIMESTAMP("2026-08-25") AND tm < TIMESTAMP("2026-08-27")
  AND state IN (3,4,5) AND cmd IN (0,1)
""")

run("orders 90d closed (volume + cost check)", """
SELECT COUNT(*) AS n_rows, COUNT(DISTINCT login) AS logins
FROM `zfx-dwh-prod.mt4_live01.orders`
WHERE _close_ts_partition >= TIMESTAMP("2026-05-29") AND _close_ts_partition < TIMESTAMP("2026-08-27")
""")

