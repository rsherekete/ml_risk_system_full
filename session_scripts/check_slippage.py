from google.cloud import bigquery
client = bigquery.Client(project="zfx-dwh-prod")

for ref in ("data_marts_multi_region.slippage_monitoring",
            "data_marts_multi_region.zeal_mt5_slippage",
            "reporting.tbl_cid_toxic"):
    print(f"=== {ref} ===")
    t = client.get_table(f"zfx-dwh-prod.{ref}")
    part = t.time_partitioning.field if t.time_partitioning else None
    print(f"  {t.num_rows:,} rows, {t.num_bytes/1e9:.1f} GB, modified {t.modified:%Y-%m-%d %H:%M}, part={part}")
    print(f"  columns: {[f.name for f in t.schema]}")
    print()

# Date coverage + server mapping for the main one
q = """
SELECT MAX(open_time) AS last_open, MIN(open_time) AS first_open, COUNT(*) AS n
FROM `zfx-dwh-prod.data_marts_multi_region.slippage_monitoring`
WHERE open_time >= TIMESTAMP('2026-08-01')
"""
try:
    job = client.query(q)
    row = list(job.result())[0]
    print(f"slippage_monitoring since 2026-08-01: {row['n']:,} rows, {row['first_open']} -> {row['last_open']}")
    print(f"[{job.total_bytes_processed/1e9:.2f} GB]")
except Exception as exc:
    print(f"date probe failed: {str(exc)[:200]}")
