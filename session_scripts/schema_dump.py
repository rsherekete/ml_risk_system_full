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
    if t.range_partitioning:
        print(f"    range_partitioning: field={t.range_partitioning.field}")
    if t.clustering_fields:
        print(f"    clustering: {t.clustering_fields}")
    if t.table_type == "VIEW" and t.view_query:
        print(f"    VIEW QUERY:\n{t.view_query}")
    print("    schema:")
    for f in t.schema:
        print(f"      {f.name:30s} {f.field_type:12s} {f.mode}")

targets = [
    ("mt4_live01", "orders"),
    ("mt4_live01", "traderecord"),
    ("mt4_live01", "ticks"),
    ("mt4_qsm01", "ticks"),
]

for ds, tbl in targets:
    dump(ds, tbl)
