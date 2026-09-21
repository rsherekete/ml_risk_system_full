from google.cloud import bigquery
client = bigquery.Client(project="zfx-dwh-prod")

q = """
SELECT server_name,
       COUNT(*) AS n,
       COUNT(DISTINCT account_number) AS accts,
       STRING_AGG(DISTINCT account_is_toxic ORDER BY account_is_toxic LIMIT 5) AS toxic_values,
       MIN(trading_date) AS first_day,
       MAX(trading_date) AS last_day,
       STRING_AGG(DISTINCT book ORDER BY book) AS books
FROM `zfx-dwh-prod.data_marts_multi_region.slippage_monitoring`
WHERE trading_date >= DATE('2026-05-29') AND trading_date < DATE('2026-08-28')
GROUP BY server_name
ORDER BY n DESC
"""
job = client.query(q)
print(f"{'server':<22}{'rows':>13}{'accts':>9}   range                    toxic values | books")
for r in job.result():
    print(f"{r['server_name']:<22}{r['n']:>13,}{r['accts']:>9,}   "
          f"{r['first_day']} -> {r['last_day']}   {r['toxic_values']} | {r['books']}")
print(f"[scanned {job.total_bytes_processed/1e9:.2f} GB]")

q2 = """
SELECT client_segment_hist, trading_profile, mtgroup_category, COUNT(*) AS n
FROM `zfx-dwh-prod.data_marts_multi_region.slippage_monitoring`
WHERE trading_date >= DATE('2026-08-20') AND trading_date < DATE('2026-08-28')
GROUP BY 1,2,3 ORDER BY n DESC LIMIT 15
"""
job2 = client.query(q2)
print("\nexisting client segmentation values (last 8 days):")
for r in job2.result():
    print(f"  segment={str(r['client_segment_hist']):<22} profile={str(r['trading_profile']):<18} "
          f"group={str(r['mtgroup_category']):<14} n={r['n']:,}")
print(f"[scanned {job2.total_bytes_processed/1e9:.2f} GB]")
