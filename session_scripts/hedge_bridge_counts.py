from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

for ds in ["mt4_live01", "mt4_live02", "mt4_live03", "mt4_live04"]:
    for tbl in ["hedge", "mt_bridge_mapping", "mt_bridge_mapping_exceptions"]:
        ref = f"zfx-dwh-prod.{ds}.{tbl}"
        try:
            t = client.get_table(ref)
            print(f"{ref}: rows={t.num_rows} bytes={t.num_bytes}")
        except Exception as e:
            print(f"{ref}: ERROR {e}")
