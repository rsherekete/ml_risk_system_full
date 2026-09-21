import sys
from google.cloud import bigquery

project_id = "zfx-dwh-prod"
client = bigquery.Client(project=project_id)

print("Auth OK, listing datasets...")
datasets = list(client.list_datasets())
print(f"{len(datasets)} datasets visible:")
for ds in datasets:
    print(f"  {ds.dataset_id}")
