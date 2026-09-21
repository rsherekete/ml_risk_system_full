from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

def show_table(full_id):
    print("=" * 90)
    print("TABLE:", full_id)
    try:
        t = client.get_table(full_id)
        print("  type:", t.table_type, " rows:", t.num_rows, " size_bytes:", t.num_bytes)
        print("  created:", t.created, " modified:", t.modified)
        print("  description:", t.description)
        print("  labels:", t.labels)
        print("  schema:")
        for f in t.schema:
            print("   -", f.name, f.field_type, f.mode)
        if t.table_type == "VIEW" and t.view_query:
            print("  VIEW QUERY:")
            print(t.view_query)
    except Exception as e:
        print("  ERROR:", e)

# Dubai tables in data_marts_multi_region
dubai_tables = [
    "mt5_dubai_live01_accounts",
    "mt5_dubai_live01_accounts_assigned_user_rel_hist",
    "mt5_dubai_live01_accounts_flagged",
    "mt5_dubai_live01_closed_trades",
    "mt5_dubai_live01_leads",
    "mt5_dubai_live01_open_trades",
    "mt5_dubai_live01_trading_accounts",
    "mt5_dubai_live01_trading_accounts_flagged",
    "mt5_dubai_live01_trading_conditions",
    "mt5_dubai_live01_transactions",
    "mt5_dubai_live01_users",
]
for t in dubai_tables:
    show_table(f"zfx-dwh-prod.data_marts_multi_region.{t}")

# For comparison: mt5_live01 core trade/account tables (already-confirmed-good server)
for t in ["deals", "dealrecord", "accounts", "orders", "orderrecord", "userrecord"]:
    show_table(f"zfx-dwh-prod.mt5_live01.{t}")

# mt5_demo01 tables
for t in ["accounts", "dailyrecord", "orderrecord", "orders", "userrecord"]:
    show_table(f"zfx-dwh-prod.mt5_demo01.{t}")

# mt4_demo_rep tables
for t in ["accounts", "orders", "userrecord"]:
    show_table(f"zfx-dwh-prod.mt4_demo_rep.{t}")
