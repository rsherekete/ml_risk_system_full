from google.cloud import bigquery
client = bigquery.Client(project="zfx-dwh-prod")

for dataset in ("markouts_v2", "topbook"):
    print(f"=== {dataset} ===")
    try:
        tables = list(client.list_tables(dataset))
        for t in tables[:15]:
            full = client.get_table(f"zfx-dwh-prod.{dataset}.{t.table_id}")
            part = full.time_partitioning.field if full.time_partitioning else None
            print(f"  {t.table_id:<40} {full.num_rows:>14,} rows  {full.num_bytes/1e9:>8.1f} GB  part={part}")
        if tables:
            sample = client.get_table(f"zfx-dwh-prod.{dataset}.{tables[0].table_id}")
            print(f"  -> {tables[0].table_id} columns: {[f.name for f in sample.schema]}")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {str(exc)[:120]}")
    print()
