from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

print("Listing all tables in operational_data_store (filtering for dubai/mt5/traze relevant)...")
tables = list(client.list_tables("operational_data_store"))
print(f"Total tables in operational_data_store: {len(tables)}")
relevant = [t for t in tables if "dubai" in t.table_id.lower() or "traze" in t.table_id.lower()]
print(f"Relevant (dubai/traze) tables: {len(relevant)}")
for t in relevant:
    print("  -", t.table_id, "(", t.table_type, ")")
