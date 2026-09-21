from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

candidates = [
    "mt5_dubai_tax_invoice",
    "mena_audit_202608",
    "mt4_demo_rep",
    "dv_raw_external_dictionaries_multi_region",
    "dwh1_source_multi_region",
    "dwh2_source_multi_region",
    "data_marts_multi_region",
    "ecn_ts_asia",
    "ecn_ts",
    "risk_monitor",
    "mt5_live01",
    "mt5_demo01",
    "mt4_demo01" if False else None,  # placeholder, doesn't exist per listing
]

for ds_id in candidates:
    if not ds_id:
        continue
    print("=" * 80)
    print("DATASET:", ds_id)
    try:
        tables = list(client.list_tables(ds_id))
        print(f"  {len(tables)} tables")
        for t in tables:
            print("   -", t.table_id, "(", t.table_type, ")")
    except Exception as e:
        print("  ERROR:", e)
