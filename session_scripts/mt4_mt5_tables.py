from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

datasets = ['mt4_demo_rep', 'mt4_live01', 'mt4_live02', 'mt4_live03', 'mt4_live04', 'mt4_qsm01',
            'mt5_demo01', 'mt5_live01', 'mt5_dubai_tax_invoice']

for ds in datasets:
    print(f"\n=== Dataset: {ds} ===")
    try:
        tables = list(client.list_tables(f"zfx-dwh-prod.{ds}"))
        print(f"  ({len(tables)} tables)")
        for t in tables:
            print(f"  {t.table_id}  (type={t.table_type})")
    except Exception as e:
        print(f"  ERROR: {e}")
