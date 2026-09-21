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
    if t.table_type == "VIEW" and t.view_query and show_view_query:
        print(f"    VIEW QUERY:\n{t.view_query}")
    cols = ", ".join(f"{f.name}:{f.field_type}" for f in t.schema)
    print(f"    columns: {cols}")

targets = [
    ("virtual_views", "mb_modeB_amt_class_daily"),
    ("virtual_views", "mb_prd_modeB_trade_30d"),
    ("views", "superapp_modeb_clients"),
    ("data_marts", "accounts"),
    ("data_marts", "trading_accounts"),
]
for ds, tbl in targets:
    dump(ds, tbl)
