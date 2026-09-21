"""Is every server actually represented, at every stage of the pipeline?"""
import sys, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.bigquery_data_client import BQ_SOURCE_FOR_DATABASE, DEMO_DATABASES

BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"

print("=== 1. Configured databases ===")
for name, spec in BQ_SOURCE_FOR_DATABASE.items():
    flag = "DEMO (excluded by design)" if name in DEMO_DATABASES else ("UNAVAILABLE: " + spec.unavailable_reason[:60] if spec.unavailable_reason else "live")
    print(f"  {name:<20} {spec.dataset}.{spec.table:<28} {flag}")

print("\n=== 2. What actually landed in the 90-day records pull ===")
records_meta = pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", columns=["database", "account_key", "timestamp"])
per_db = records_meta.groupby("database", observed=True).agg(
    rows=("account_key", "size"), accounts=("account_key", "nunique"),
    first=("timestamp", "min"), last=("timestamp", "max"),
)
print(per_db.to_string())
print(f"  TOTAL {len(records_meta):,} rows, {records_meta['account_key'].nunique():,} accounts")

print("\n=== 3. Markout coverage per database ===")
markouts = pd.read_parquet(f"{BASE}\\markout_by_account_day.parquet")
mk = markouts.groupby("database", observed=True).agg(
    account_days=("account_key", "size"), accounts=("account_key", "nunique"),
)
print(mk.to_string())

records_accounts = records_meta.groupby("database", observed=True)["account_key"].nunique()
markout_accounts = markouts.groupby("database", observed=True)["account_key"].nunique()
print("\n  coverage by database (markout accounts / record accounts):")
for database in records_accounts.index:
    have = int(markout_accounts.get(database, 0))
    total = int(records_accounts[database])
    status = "OK" if have > 0 else "*** NO MARKOUT DATA ***"
    print(f"    {database:<20} {have:>7,} / {total:>7,}  {have/max(total,1):>6.1%}  {status}")

covered = set(markout_accounts.index)
missing = [d for d in records_accounts.index if d not in covered]
if missing:
    print(f"\n  GAP: no market context at all for {missing}")
    lost = int(records_accounts[missing].sum())
    print(f"  that is {lost:,} accounts ({lost/records_accounts.sum():.1%} of the book) with zero market context")
