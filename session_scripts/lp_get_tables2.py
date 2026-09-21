from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

targets = [
    ("topbook", "topbook_2023"),
    ("topbook", "topbook_dq_bid_ask"),
    ("topbook", "spread_target_v1"),
    ("topbook", "price_delay_seconds"),
    ("ecn_ts", "Common"),
]

for ds, tbl in targets:
    print(f"\n{'='*70}\n{ds}.{tbl}\n{'='*70}")
    try:
        table = client.get_table(f"zfx-dwh-prod.{ds}.{tbl}")
    except Exception as e:
        print(f"  ERROR: {e}")
        continue
    print(f"  table_type: {table.table_type}")
    print(f"  num_rows: {table.num_rows}  num_bytes: {table.num_bytes}")
    print(f"  created: {table.created}  modified: {table.modified}")
    if table.time_partitioning:
        tp = table.time_partitioning
        print(f"  time_partitioning: type={tp.type_} field={tp.field}")
    if table.clustering_fields:
        print(f"  clustering_fields: {table.clustering_fields}")
    print(f"  schema ({len(table.schema)} fields):")
    for f in table.schema:
        print(f"    - {f.name}: {f.field_type} ({f.mode})")

# Check hive partitioning config on bank_quotes external tables
print(f"\n{'='*70}\nHive partitioning check on bank_quotes external tables\n{'='*70}")
for tbl in ["bank_quotes", "bank_quotes_gold", "bank_quotes_dbs04p", "bank_quotes_6"]:
    table = client.get_table(f"zfx-dwh-prod.bank_quotes.{tbl}")
    edc = table.external_data_configuration
    print(f"{tbl}: hive_partitioning={edc.hive_partitioning}")
