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
    ("ecn_ts", "DoneTrades"),
    ("ecn_ts", "NetPositions"),
    ("ecn_ts", "NetPositionSnapshots"),
    ("ecn_ts", "WorkingSessions"),
    ("ecn_reports", "BankExecutionReports"),
    ("ecn_reports", "ExecutionReports"),
    ("ecn_reports", "BankOrders"),
]
for ds, tbl in targets:
    dump(ds, tbl)
