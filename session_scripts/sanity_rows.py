import sys, gc, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.bigquery_data_client import compact_memory
from trading_data.behaviour_features import daily_behaviour_features

PATH = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"

parts = []
for database in sorted(pd.read_parquet(PATH, columns=["database"])["database"].unique()):
    part = compact_memory(pd.read_parquet(PATH, filters=[("database", "==", database)]))
    parts.append(daily_behaviour_features(part))
    del part; gc.collect()
daily = pd.concat(parts, ignore_index=True)
del parts; gc.collect()

per_account = daily.groupby("account_key", observed=True).agg(
    active_days=("day", "nunique"), first=("day", "min"), last=("day", "max"),
)
per_account["span_days"] = (per_account["last"] - per_account["first"]).dt.days + 1
per_account["density"] = per_account["active_days"] / per_account["span_days"]

print(f"accounts: {len(per_account):,}")
print(f"total account-day rows: {len(daily):,}")
print(f"rows after dropping each account's LAST day (no next-active label): {len(daily) - len(per_account):,}")
print("\nactive days per account (over the 90-day window):")
print(per_account["active_days"].describe(percentiles=[.1,.25,.5,.75,.9,.99]).to_string())
print("\nspan (first->last active day) per account:")
print(per_account["span_days"].describe(percentiles=[.1,.5,.9]).to_string())
print("\ntrading density (active days / span) -- 1.0 = trades every day they're present:")
print(per_account["density"].describe(percentiles=[.1,.5,.9]).to_string())
print(f"\naccounts active on only 1 day:  {(per_account['active_days']==1).sum():,} "
      f"({(per_account['active_days']==1).mean():.1%}) -- these contribute ZERO labelled rows")
print(f"accounts with >=6 active days:  {(per_account['active_days']>=6).sum():,} "
      f"({(per_account['active_days']>=6).mean():.1%}) -- these have full lag-5 coverage")
print(f"accounts spanning >=80 days:    {(per_account['span_days']>=80).sum():,} "
      f"({(per_account['span_days']>=80).mean():.1%})")
