"""What does two years of history cost to extract, and how big is it?

Dry-run only -- no data is read and nothing is billed. Establishes the price
before committing, because the 90-day extract is 0.87 GB and a naive 8x
extrapolation could be expensive enough to matter.
"""
import sys

from google.cloud import bigquery

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from trading_data.bigquery_data_client import BQ_SOURCE_FOR_DATABASE

client = bigquery.Client(project="zfx-dwh-prod")
dry = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

WINDOWS = {"90 days": 90, "1 year": 365, "2 years": 730}
PRICE_PER_TB = 6.25

print(f"{'database':<22}{'table':<52}" + "".join(f"{k:>14}" for k in WINDOWS))
totals = {k: 0 for k in WINDOWS}

for database, source in sorted(BQ_SOURCE_FOR_DATABASE.items()):
    if getattr(source, "unavailable_reason", None):
        print(f"{database:<22}{'-- unavailable --':<52}")
        continue
    table = f"zfx-dwh-prod.{source.dataset}.{source.table}"
    try:
        meta = client.get_table(table)
    except Exception as error:
        print(f"{database:<22}{table[-50:]:<52}  ERROR {type(error).__name__}")
        continue
    partition_field = meta.time_partitioning.field if meta.time_partitioning else None
    row = f"{database:<22}{table[-50:]:<52}"
    for label, days in WINDOWS.items():
        if partition_field:
            sql = (f"SELECT COUNT(*) FROM `{table}` WHERE {partition_field} >= "
                   f"TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)")
        else:
            sql = f"SELECT COUNT(*) FROM `{table}`"
        try:
            scanned = client.query(sql, job_config=dry).total_bytes_processed
        except Exception:
            scanned = 0
        totals[label] += scanned
        row += f"{scanned / 1e9:>12.1f}GB"
    print(row, flush=True)

print(f"\n{'TOTAL':<74}" + "".join(f"{totals[k] / 1e9:>12.1f}GB" for k in WINDOWS))
print(f"{'estimated cost':<74}" +
      "".join(f"{'$' + format(totals[k] / 1e12 * PRICE_PER_TB, '.2f'):>14}" for k in WINDOWS))
print("\nNote: a COUNT(*) dry run reports the bytes the partition filter admits, which is")
print("the right proxy for a SELECT of the same range. Selecting fewer columns costs less.")
