from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

def dump(dataset, table):
    ref = f"zfx-dwh-prod.{dataset}.{table}"
    try:
        t = client.get_table(ref)
    except Exception as e:
        print(f"\n### {ref}  ERROR: {e}")
        return
    print(f"\n### {ref}  type={t.table_type}  rows={t.num_rows}  bytes={t.num_bytes}")
    if t.time_partitioning:
        print(f"    partitioning: field={t.time_partitioning.field} type={t.time_partitioning.type_}")
    if t.clustering_fields:
        print(f"    clustering: {t.clustering_fields}")
    if t.table_type == "VIEW" and t.view_query:
        print(f"    VIEW QUERY:\n{t.view_query}")
    print("    schema:")
    for f in t.schema:
        print(f"      {f.name:30s} {f.field_type:12s} {f.mode}")

targets = [
    ("operational_data_store", "mt4_live01__traderecord"),
    ("mt4_live01", "hedge"),
    ("mt4_live01", "mt_bridge_mapping"),
    ("mt5_live01", "dealrecord"),
    ("mt5_live01", "deals"),
    ("mt5_live01", "orderrecord"),
    ("mt5_live01", "orders"),
    ("mt5_live01", "positionrecord"),
    ("mt5_live01", "positions"),
    ("mt5_live01", "transactions"),
    ("mt5_live01", "ticks"),
    ("mt5_live01", "gateways"),
    ("mt5_live01", "gateway_translates"),
]

for ds, tbl in targets:
    dump(ds, tbl)
