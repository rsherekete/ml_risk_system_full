from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

datasets = list(client.list_datasets())
print(f"Scanning {len(datasets)} datasets for table names matching deal/demo/dubai...\n")

keywords = ["deal", "demo", "dubai"]

hits = []
errors = []
for ds in datasets:
    ds_id = ds.dataset_id
    try:
        tables = list(client.list_tables(f"zfx-dwh-prod.{ds_id}"))
        for t in tables:
            tid_lower = t.table_id.lower()
            if any(k in tid_lower for k in keywords):
                hits.append((ds_id, t.table_id))
    except Exception as e:
        errors.append((ds_id, str(e)))

print("=== MATCHES (table name contains deal/demo/dubai) ===")
for ds_id, tid in hits:
    print(f"  {ds_id}.{tid}")

print(f"\nTotal matches: {len(hits)}")
print(f"\nErrors encountered on {len(errors)} datasets:")
for ds_id, err in errors[:20]:
    print(f"  {ds_id}: {err[:150]}")
