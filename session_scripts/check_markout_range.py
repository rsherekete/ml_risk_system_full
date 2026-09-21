from google.cloud import bigquery
client = bigquery.Client(project="zfx-dwh-prod")

t = client.get_table("zfx-dwh-prod.markouts_v2.markouts")
print(f"markouts: {t.num_rows:,} rows, {t.num_bytes/1e9:.1f} GB")
print(f"created {t.created}, last modified {t.modified}")
print(f"partitioning: {t.time_partitioning}")

q = """
SELECT MIN(close_time_ms) AS first_ts, MAX(close_time_ms) AS last_ts,
       COUNT(DISTINCT server_name) AS servers
FROM `zfx-dwh-prod.markouts_v2.markouts`
"""
job = client.query(q)
row = list(job.result())[0]
print(f"\nactual data range: {row['first_ts']} -> {row['last_ts']}  ({row['servers']} servers)")
print(f"[scanned {job.total_bytes_processed/1e9:.2f} GB]")

q2 = """
SELECT server_name, COUNT(*) AS n, MAX(DATE(close_time)) AS last_day
FROM `zfx-dwh-prod.markouts_v2.markouts`
GROUP BY server_name ORDER BY n DESC LIMIT 20
"""
job2 = client.query(q2)
print("\nservers present:")
for r in job2.result():
    print(f"  {r['server_name']:<30} {r['n']:>12,}  last={r['last_day']}")
print(f"[scanned {job2.total_bytes_processed/1e9:.2f} GB]")
