from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

cluster = ['ecn_ts', 'ecn_ts_asia', 'ecn_configuration', 'ecn_reports', 'bank_quotes',
           'bankquotes_sandbox', 'mysql_bankquotes', 'exness_ticks_from_site', 'topbook', 'markouts_v2']

for ds_id in cluster:
    print(f"\n=== DATASET: {ds_id} ===")
    ds_ref = client.dataset(ds_id)
    try:
        ds = client.get_dataset(ds_ref)
        print(f"  description: {ds.description!r}")
        print(f"  location: {ds.location}")
        labels = ds.labels
        print(f"  labels: {labels}")
    except Exception as e:
        print(f"  ERROR getting dataset: {e}")
        continue
    try:
        tables = list(client.list_tables(ds_ref))
        print(f"  num tables/views: {len(tables)}")
        for t in tables:
            print(f"    - {t.table_id} (type={t.table_type})")
    except Exception as e:
        print(f"  ERROR listing tables: {e}")
