from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

tables = list(client.list_tables("operational_data_store"))
demo_related = [t for t in tables if "demo" in t.table_id.lower()]
print(f"demo-related tables in operational_data_store: {len(demo_related)}")
for t in demo_related:
    print("  -", t.table_id, "(", t.table_type, ")")

# also check data_marts_multi_region for demo-prefixed marts
tables2 = list(client.list_tables("data_marts_multi_region"))
demo2 = [t for t in tables2 if "demo" in t.table_id.lower()]
print(f"\ndemo-related tables in data_marts_multi_region: {len(demo2)}")
for t in demo2:
    print("  -", t.table_id)

# check dataset-level description/labels for mt5_demo01, mt4_demo_rep, mt5_dubai_live01(doesn't exist), dubai/mena named datasets
for ds in ["mt5_demo01", "mt4_demo_rep", "mt5_dubai_tax_invoice", "mena_audit_202608", "data_marts_multi_region", "operational_data_store"]:
    d = client.get_dataset(ds)
    print(f"\nDATASET {ds}: description={d.description!r} labels={d.labels}")
