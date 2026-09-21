"""Compare walk-forward training regimes on the real 90-day BigQuery pull.

Processes ONE database at a time: the 33.5M-row combined frame does not fit in
memory, but each database's slice does, and `daily_account_features` collapses
it to a few hundred thousand rows before anything else has to hold it.
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import pandas as pd
import pyarrow.parquet as pq
from trading_data.research import compare_training_windows, daily_account_features, supervised_dataset
from trading_data.bigquery_data_client import compact_memory

path = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"
needed = ["account_key", "database", "platform", "region", "timestamp", "symbol",
          "cmd", "state", "volume_lots", "profit", "net_profit"]

parquet_file = pq.ParquetFile(path)
databases = sorted(pd.read_parquet(path, columns=["database"])["database"].unique())
print(f"databases: {databases}")

feature_frames = []
for database in databases:
    t0 = time.time()
    slice_frame = pd.read_parquet(path, columns=needed, filters=[("database", "==", database)])
    slice_frame = compact_memory(slice_frame)
    features = daily_account_features(slice_frame)
    feature_frames.append(features)
    print(f"  {database:<18} {len(slice_frame):>10,} rows -> {len(features):>8,} feature rows  ({time.time()-t0:.0f}s)")
    del slice_frame
    gc.collect()

features = pd.concat(feature_frames, ignore_index=True)
del feature_frames
gc.collect()
dataset = supervised_dataset(features)
del features
gc.collect()

feature_columns = [c for c in dataset.columns if c not in {"account_key", "database", "platform", "decision_day", "target_profit", "target_loss"}]
for column in feature_columns:
    dataset[column] = pd.to_numeric(dataset[column], errors="coerce").astype("float32")
dataset["target_profit"] = pd.to_numeric(dataset["target_profit"], errors="coerce").astype("float64")
dataset["target_loss"] = dataset["target_loss"].astype(bool)
print(f"\ndataset: {len(dataset):,} rows, {dataset['account_key'].nunique():,} accounts, "
      f"{pd.to_datetime(dataset['decision_day']).dt.normalize().nunique()} days, "
      f"{dataset.memory_usage(deep=True).sum()/1e6:.0f} MB")

print("\n=== CLASSIFICATION: which training regime ranks losses best? ===")
t0 = time.time()
print(compare_training_windows(dataset, task="classification").to_string(index=False))
print(f"[{time.time()-t0:.0f}s]")

print("\n=== REGRESSION (drives routing): which regime ranks dollars best? ===")
t0 = time.time()
print(compare_training_windows(dataset, task="regression").to_string(index=False))
print(f"[{time.time()-t0:.0f}s]")
