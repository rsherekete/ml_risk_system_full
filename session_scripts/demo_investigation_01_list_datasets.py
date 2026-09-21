from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")

print("=== ALL DATASETS ===")
datasets = list(client.list_datasets())
for ds in sorted(d.dataset_id for d in datasets):
    print(ds)
print(f"\nTotal datasets: {len(datasets)}")
