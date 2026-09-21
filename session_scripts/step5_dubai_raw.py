from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

def show_table(full_id, schema_only=False):
    print("=" * 90)
    print("TABLE:", full_id)
    try:
        t = client.get_table(full_id)
        print("  type:", t.table_type, " rows:", t.num_rows, " size_bytes:", t.num_bytes)
        print("  created:", t.created, " modified:", t.modified)
        print("  partitioning:", t.time_partitioning, t.range_partitioning)
        print("  clustering_fields:", t.clustering_fields)
        print("  schema (%d fields):" % len(t.schema))
        for f in t.schema:
            print("   -", f.name, f.field_type, f.mode)
    except Exception as e:
        print("  ERROR:", e)

raw_dubai = [
    "dubai_traze_mt5_dubai_live01_deals",
    "dubai_traze_mt5_dubai_live01_dealrecord",
    "dubai_traze_mt5_dubai_live01_orders",
    "dubai_traze_mt5_dubai_live01_orderrecord",
    "dubai_traze_mt5_dubai_live01_positions",
    "dubai_traze_mt5_dubai_live01_positionrecord",
    "dubai_traze_mt5_dubai_live01_userrecord",
    "dubai_traze_mt5_dubai_live01_accounts",
    "dubai_traze_mt5_dubai_live01_groups",
    "dubai_traze_mt5_dubai_live01_symbols",
    "dubai_traze_mt5_dubai_live01_loginrecord",
]
for t in raw_dubai:
    show_table(f"zfx-dwh-prod.operational_data_store.{t}")

print()
print("#### COMPARISON: mt5_live01 raw tables (confirmed-good baseline) ####")
for t in ["deals", "dealrecord", "orders", "orderrecord", "positions", "positionrecord", "userrecord", "accounts", "groups", "symbols", "loginrecord"]:
    show_table(f"zfx-dwh-prod.mt5_live01.{t}")
