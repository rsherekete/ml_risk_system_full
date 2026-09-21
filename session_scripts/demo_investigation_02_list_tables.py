from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

datasets_to_check = [
    "mt4_demo_rep",
    "mt5_demo01",
    "mam_live01",
    "mam_live02",
    "mam_live03",
    "mam_live04",
    "fbr_live01",
    "fbr_live02",
    "fbr_live03",
    "fbr_live04",
    "fbr_live01_jrnl",
    "mt4_qsm01",
    "mt5_dubai_tax_invoice",
    "mt4_live01",  # reference: known-good live server for comparison
    "mt5_live01",  # reference: known-good live server for comparison
]

for ds_id in datasets_to_check:
    print(f"\n=== DATASET: {ds_id} ===")
    try:
        tables = list(client.list_tables(f"zfx-dwh-prod.{ds_id}"))
        for t in sorted(tbl.table_id for tbl in tables):
            print(f"  {t}")
        print(f"  ({len(tables)} tables)")
    except Exception as e:
        print(f"  ERROR: {e}")
