from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

targets = {
    'mt5_demo01': ['orderrecord_temp', 'orders_temp', 'userrecord_temp'],
    'mt5_live01': [
        'dailyrecord_temp', 'deals_temp', 'loginrecord_temp', 'orderrecord_temp',
        'orders_temp', 'positionrecord_temp', 'server_logs_temp', 'server_logs_temp_20260825_a403a6'
    ],
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
        print(f"columns: {[f.name for f in t.schema]}")
