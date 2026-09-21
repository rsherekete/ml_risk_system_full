from google.cloud import bigquery
client = bigquery.Client(project="zfx-dwh-prod")
t = client.get_table("zfx-dwh-prod.mt4_demo_rep.orders")
print("partitioning:", t.time_partitioning)
print("rows:", f"{t.num_rows:,}")
print("columns:", [f.name for f in t.schema])
