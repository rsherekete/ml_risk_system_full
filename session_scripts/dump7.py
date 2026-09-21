from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

def dump(dataset, table, show_view_query=True):
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
    if t.table_type == "VIEW" and t.view_query and show_view_query:
        print(f"    VIEW QUERY:\n{t.view_query}")
    cols = ", ".join(f"{f.name}:{f.field_type}" for f in t.schema)
    print(f"    columns: {cols}")

targets = [
    ("markouts_v2", "markouts"),
    ("markouts_v2", "markouts_order_close"),
    ("markouts_v2", "markouts_order_open"),
    ("data_marts", "markouts"),
    ("data_marts", "zfx_exposure_monitor"),
    ("data_marts", "risk_backtesting"),
    ("data_marts", "trust_model_scoring_result"),
    ("data_marts", "ticks"),
    ("data_marts", "qsm_ticks"),
    ("topbook", "topbook"),
    ("fbr_live01", "TSSessions"),
]
for ds, tbl in targets:
    dump(ds, tbl)

print("\n=== ecn_reports tables ===")
for t in client.list_tables("zfx-dwh-prod.ecn_reports"):
    print(f"  {t.table_id} (type={t.table_type})")

print("\n=== ecn_ts tables ===")
for t in client.list_tables("zfx-dwh-prod.ecn_ts"):
    print(f"  {t.table_id} (type={t.table_type})")

print("\n=== ecn_ts_asia tables ===")
for t in client.list_tables("zfx-dwh-prod.ecn_ts_asia"):
    print(f"  {t.table_id} (type={t.table_type})")
