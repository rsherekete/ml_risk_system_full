from google.cloud import bigquery
import sys

client = bigquery.Client(project='zfx-dwh-prod')

cluster = ['ecn_ts', 'ecn_ts_asia', 'ecn_configuration', 'ecn_reports', 'bank_quotes',
           'bankquotes_sandbox', 'mysql_bankquotes', 'exness_ticks_from_site', 'topbook', 'markouts_v2']

# First list all datasets in the project to confirm exact names / find close matches
print("=== ALL DATASETS IN PROJECT ===")
all_ds = list(client.list_datasets())
all_ds_ids = [d.dataset_id for d in all_ds]
for d in all_ds_ids:
    print(d)
print(f"Total datasets: {len(all_ds_ids)}")

print()
print("=== CLUSTER DATASET MATCH CHECK ===")
for c in cluster:
    matches = [d for d in all_ds_ids if c.lower() in d.lower() or d.lower() in c.lower()]
    print(f"{c}: exact_match={c in all_ds_ids}, close_matches={matches}")
