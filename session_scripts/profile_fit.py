"""Where does the walk-forward time actually go? Measure, don't guess.

LightGBM on 500k x 174 float32 should fit in seconds. An arm taking ~an hour
means the cost is somewhere else -- most likely the per-iteration pandas work
(boolean .loc slicing copies the whole matrix) rather than the boosting itself.
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from trading_data.behaviour_features import build_active_day_frame, feature_columns
from trading_data.bigquery_data_client import compact_memory

BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"

t0 = time.time()
parts = []
for database in sorted(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", columns=["database"])["database"].unique()):
    part = compact_memory(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", filters=[("database", "==", database)]))
    parts.append(build_active_day_frame(part, max_gap_days=None))
    del part; gc.collect()
frame = pd.concat(parts, ignore_index=True)
del parts; gc.collect()
print(f"load+build: {time.time()-t0:.0f}s | frame {frame.shape} | {frame.memory_usage(deep=True).sum()/1e9:.2f} GB")

columns = feature_columns(frame)
t0 = time.time()
for column in columns:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
print(f"coerce {len(columns)} cols: {time.time()-t0:.0f}s")
print(f"dtypes across whole frame: {frame.dtypes.astype(str).value_counts().to_dict()}")

frame["decision_day"] = pd.to_datetime(frame["decision_day"]).dt.normalize()
frame["pnl"] = pd.to_numeric(frame["target_profit"], errors="coerce")
frame = frame.loc[frame["pnl"].notna()].reset_index(drop=True)
frame["label_wins"] = frame["pnl"] > 0
days = sorted(frame["decision_day"].unique())

# Worst-case slice: the final refit, training on ~all prior days.
offset = len(days) - 1
t0 = time.time()
train = frame["decision_day"].isin(days[:offset])
t_isin = time.time() - t0

t0 = time.time()
X = frame.loc[train, columns]
t_slice = time.time() - t0

t0 = time.time()
X_np = np.ascontiguousarray(frame.loc[train, columns].to_numpy(dtype="float32"))
t_numpy = time.time() - t0

y = frame.loc[train, "label_wins"]
print(f"\nlargest training slice: {X.shape}")
print(f"  isin mask          {t_isin:>7.1f}s")
print(f"  .loc[mask, cols]   {t_slice:>7.1f}s   <- copies the whole matrix, EVERY refit")
print(f"  -> contiguous f32  {t_numpy:>7.1f}s")

for label, data in (("DataFrame", X), ("numpy f32", X_np)):
    for jobs in (-1, 6):
        model = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, subsample=0.5, subsample_freq=1,
                                   colsample_bytree=0.5, n_jobs=jobs, verbose=-1, random_state=0)
        t0 = time.time()
        model.fit(data, y)
        print(f"  fit({label:<9}, n_jobs={jobs:>2}) {time.time()-t0:>7.1f}s", flush=True)
