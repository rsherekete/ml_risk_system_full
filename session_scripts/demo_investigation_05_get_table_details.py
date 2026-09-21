from google.cloud import bigquery
from datetime import datetime, timezone

client = bigquery.Client(project="zfx-dwh-prod")

tables_to_inspect = [
    "mt4_demo_rep.orders",
    "mt4_demo_rep.accounts",
    "mt4_demo_rep.userrecord",
    "mt5_demo01.orderrecord",
    "mt5_demo01.orders",
    "mt5_demo01.dailyrecord",
    "mt5_demo01.accounts",
    "mt5_demo01.userrecord",
    "reporting.tbl_mirror_recent_trades_demo",
    "virtual_views.demo_acc_analysis",
    "operational_data_store.dubai_traze_mt5_dubai_live01_deals",
    "operational_data_store.dubai_traze_mt5_dubai_live01_dealrecord",
    "operational_data_store.dubai_traze_mt5_dubai_live01_orderrecord",
    "operational_data_store.dubai_traze_mt5_dubai_live01_orders",
    "operational_data_store.dubai_traze_mt5_dubai_live01_accounts",
    "operational_data_store.dubai_traze_mt5_dubai_live01_userrecord",
    "mt5_live01.deals",
    "mt5_live01.dealrecord",
]

for full in tables_to_inspect:
    ds_id, tbl_id = full.split(".")
    print(f"\n{'='*80}\n=== {full} ===")
    try:
        t = client.get_table(f"zfx-dwh-prod.{ds_id}.{tbl_id}")
        print(f"  num_rows: {t.num_rows}")
        print(f"  num_bytes: {t.num_bytes}")
        mod = t.modified
        print(f"  modified: {mod}  (days ago: {(datetime.now(timezone.utc)-mod).days if mod else 'n/a'})")
        print(f"  time_partitioning: {t.time_partitioning}")
        print(f"  clustering_fields: {t.clustering_fields}")
        print(f"  table_type: {t.table_type}")
        print(f"  Schema fields ({len(t.schema)}):")
        for f in t.schema:
            print(f"    {f.name:35s} {f.field_type:12s} {f.mode}")
    except Exception as e:
        print(f"  ERROR: {e}")
