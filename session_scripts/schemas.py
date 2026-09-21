from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

targets = {
    'mt5_demo01': ['accounts', 'dailyrecord', 'orderrecord', 'orders', 'userrecord'],
    'mt5_live01': [
        'accounts', 'dealrecord', 'deals', 'deals__backup_20260729', 'ticks',
        'orders', 'orderrecord', 'positions', 'positionrecord', 'transactions',
        'symbols', 'group_symbols', 'groups', 'commissions', 'commission_tiers',
        'concommission', 'concommtier', 'consymbol', 'consymbolsession',
        'dailyrecord', 'eodrecord', 'gateways', 'gateway_translates',
        'holidays', 'holiday_symbols', 'loginrecord', 'requestrecord',
        'server_logs', 'symbol_sessions', 'userrecord'
    ],
    'mt5_dubai_tax_invoice': ['mt5_dubai_tax_invoice_upload_history'],
}

for ds, tables in targets.items():
    print(f"\n\n########## DATASET: {ds} ##########")
    for tname in tables:
        full = f"zfx-dwh-prod.{ds}.{tname}"
        try:
            t = client.get_table(full)
        except Exception as e:
            print(f"\n--- {tname}: ERROR {e}")
            continue
        print(f"\n--- TABLE: {tname} ---")
        print(f"num_rows={t.num_rows}  num_bytes={t.num_bytes}  created={t.created}  modified={t.modified}")
        if t.time_partitioning:
            print(f"time_partitioning: field={t.time_partitioning.field} type={t.time_partitioning.type_}")
        if t.range_partitioning:
            print(f"range_partitioning: field={t.range_partitioning.field}")
        if t.clustering_fields:
            print(f"clustering_fields: {t.clustering_fields}")
        print("schema:")
        for f in t.schema:
            print(f"  {f.name:30s} {f.field_type:12s} {f.mode}")
