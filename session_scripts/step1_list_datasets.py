from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

datasets = list(client.list_datasets())
print(f"Total datasets: {len(datasets)}")
for d in datasets:
    print(d.dataset_id)
