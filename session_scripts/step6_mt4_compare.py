from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

print("### mt4_live01 tables ###")
for t in client.list_tables("mt4_live01"):
    print("  -", t.table_id, "(", t.table_type, ")")

def full_schema(full_id):
    print("=" * 80)
    print("TABLE:", full_id)
    t = client.get_table(full_id)
    print("  rows:", t.num_rows, " size:", t.num_bytes, " modified:", t.modified)
    for f in t.schema:
        print("   -", f.name, f.field_type, f.mode)

full_schema("zfx-dwh-prod.mt4_demo_rep.orders")
print()
full_schema("zfx-dwh-prod.mt4_live01.orders") if any(t.table_id=="orders" for t in client.list_tables("mt4_live01")) else print("no 'orders' table in mt4_live01")
