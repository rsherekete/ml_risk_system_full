from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

for ds_id in ["fbr_live01", "mam_live01", "risk_monitor", "ecn_ts", "ecn_reports", "topbook", "markouts_v2", "trust_model"]:
    ds = client.get_dataset(f"zfx-dwh-prod.{ds_id}")
    print(f"{ds_id}: description={ds.description!r}  labels={dict(ds.labels) if ds.labels else {}}")

print()
for dataset, table in [("fbr_live01", "MTTrades"), ("fbr_live01", "TSTrades"), ("fbr_live01", "OrderRoutes"),
                        ("fbr_live01", "Toxics"), ("fbr_live01", "GlobalSettings"), ("fbr_live01", "tbl_fx_robotron"),
                        ("ecn_ts", "DoneTrades")]:
    t = client.get_table(f"zfx-dwh-prod.{dataset}.{table}")
    print(f"{dataset}.{table}: table.description={t.description!r}")
    for f in t.schema[:5]:
        if f.description:
            print(f"    field {f.name}: {f.description!r}")
