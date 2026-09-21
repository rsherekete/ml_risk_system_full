import sys, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.bigquery_data_client import BQ_SOURCE_FOR_DATABASE, BigQueryDataClient, BigQueryUnavailableError, is_demo_database
from trading_data.research import add_provenance

day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start, end = (day - pd.Timedelta(days=90)).to_pydatetime(), (day + pd.Timedelta(days=1)).to_pydatetime()

frames, total_t0 = [], time.time()
for database in BQ_SOURCE_FOR_DATABASE:
    if is_demo_database(database):
        continue
    try:
        c = BigQueryDataClient(database)
        t0 = time.time()
        f = add_provenance(c.get_trade_records(start, end), database)
        frames.append(f)
        closed = f["state"].isin(["closed", "closed_part", "closed_by"]).sum()
        print(f"{database:<18} {len(f):>10,} rows  {f['account_key'].nunique():>6,} accts  "
              f"{closed:>9,} closed  ({time.time()-t0:.1f}s)")
    except BigQueryUnavailableError as e:
        print(f"{database:<18} skipped: {str(e)[:70]}")
    except Exception as e:
        print(f"{database:<18} FAILED {type(e).__name__}: {str(e)[:100]}")

records = pd.concat(frames, ignore_index=True)
print(f"\nTOTAL 90d: {len(records):,} rows, {records['account_key'].nunique():,} accounts, "
      f"{records.memory_usage(deep=True).sum()/1e9:.2f} GB RAM, {time.time()-total_t0:.1f}s")
print(f"day range: {records['timestamp'].min()} -> {records['timestamp'].max()}")
records.to_parquet(r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet", index=False)
print("saved to bq_90d_records.parquet for the ML-quality evaluation")
