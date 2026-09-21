from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

tables = list(client.list_tables("zfx-dwh-prod.operational_data_store"))
print(f"Total tables in operational_data_store: {len(tables)}\n")

# derive distinct prefixes (first two underscore-joined tokens) to see what "sources" are mirrored
prefixes = {}
for t in tables:
    parts = t.table_id.split("_")
    # try to find a "prefix" like dubai_traze, indonesia_traze, mt5_..., mt4_...
    prefix2 = "_".join(parts[:2])
    prefixes.setdefault(prefix2, []).append(t.table_id)

print("=== Distinct 2-token prefixes in operational_data_store ===")
for p in sorted(prefixes.keys()):
    print(f"  {p}  (n={len(prefixes[p])})")

print("\n=== Full table list ===")
for t in sorted(tbl.table_id for tbl in tables):
    print(f"  {t}")
