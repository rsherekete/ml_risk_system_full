from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

for ref in ["mt4_live01.traderecord", "operational_data_store.mt4_live01__traderecord", "mt5_live01.deals"]:
    table = client.get_table(f"zfx-dwh-prod.{ref}")
    print(f"=== {ref} ===")
    print(f"type: {table.table_type}")
    if table.view_query:
        print(f"view_query: {table.view_query}")
    tp = table.time_partitioning
    rp = table.range_partitioning
    print(f"time_partitioning: {tp}")
    print(f"range_partitioning: {rp}")
    print("schema:")
    for field in table.schema:
        print(f"  {field.name:<20} {field.field_type:<12} {field.mode}")
    print()
