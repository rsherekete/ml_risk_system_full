from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

datasets = [
    'fbr_live01', 'fbr_live01_jrnl', 'fbr_live02', 'fbr_live02_jrnl',
    'fbr_live03', 'fbr_live03_jrnl', 'fbr_live04', 'fbr_live04_jrnl',
    'mam_live01', 'mam_live02', 'mam_live03', 'mam_live04',
    'risk_monitor', 'trading', 'trading_settings_api',
    'warehouse', 'data_marts', 'views', 'virtual_views',
    'topbook', 'markouts_v2', 'exness_ticks_from_site',
]

for ds in datasets:
    print(f"\n=== Dataset: {ds} ===")
    try:
        tables = list(client.list_tables(f"zfx-dwh-prod.{ds}"))
        print(f"  ({len(tables)} tables/views)")
        for t in tables:
            print(f"  {t.table_id}  (type={t.table_type})")
    except Exception as e:
        print(f"  ERROR: {e}")
