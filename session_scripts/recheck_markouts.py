from google.cloud import bigquery
client = bigquery.Client(project="zfx-dwh-prod")

# 1. All three markout tables, not just the one checked before.
for name in ("markouts", "markouts_order_close", "markouts_order_open"):
    ref = f"zfx-dwh-prod.markouts_v2.{name}"
    try:
        t = client.get_table(ref)
        part = t.time_partitioning.field if t.time_partitioning else None
        print(f"{name:<24} {t.num_rows:>13,} rows  modified={t.modified:%Y-%m-%d %H:%M}  part={part}")
    except Exception as exc:
        print(f"{name:<24} FAILED {type(exc).__name__}")

# 2. Max date per table, using each one's own partition column.
for name, col in (("markouts", "close_time_ms"),
                  ("markouts_order_close", "close_time_ms"),
                  ("markouts_order_open", "open_time_ms")):
    try:
        job = client.query(f"SELECT MAX({col}) AS last_ts, COUNT(*) AS n FROM `zfx-dwh-prod.markouts_v2.{name}`")
        row = list(job.result())[0]
        print(f"  {name:<24} max({col}) = {row['last_ts']}   n={row['n']:,}  [{job.total_bytes_processed/1e9:.2f} GB]")
    except Exception as exc:
        print(f"  {name:<24} FAILED {str(exc)[:90]}")

# 3. Is there a NEWER markout dataset elsewhere in the project?
print("\nDatasets whose name hints at markouts / execution quality:")
for ds in client.list_datasets():
    lowered = ds.dataset_id.lower()
    if any(k in lowered for k in ("markout", "execution", "quality", "slippage", "toxic", "flow")):
        print(f"  {ds.dataset_id}")

# 4. Any markout-like TABLE in the other analytics datasets?
print("\nMarkout-like tables in analytics datasets:")
for dataset in ("data_marts", "data_products", "reporting", "risk_monitor", "trading",
                "data_marts_multi_region", "operational_data_store", "ecn_reports", "dwh1_source_multi_region"):
    try:
        for t in client.list_tables(dataset):
            lowered = t.table_id.lower()
            if any(k in lowered for k in ("markout", "slippage", "execution_quality", "toxic")):
                full = client.get_table(f"zfx-dwh-prod.{dataset}.{t.table_id}")
                print(f"  {dataset}.{t.table_id:<45} {full.num_rows:>12,} rows  modified={full.modified:%Y-%m-%d}")
    except Exception:
        pass
