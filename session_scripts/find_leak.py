"""AUC 1.0 means a feature is carrying the label. Find which one."""
import sys, gc, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from trading_data.behaviour_features import build_active_day_frame, feature_columns
from trading_data.bigquery_data_client import compact_memory

PATH = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"

# One database is plenty to expose a leak.
part = compact_memory(pd.read_parquet(PATH, filters=[("database", "==", "mt4_live03")]))
frame = build_active_day_frame(part, max_gap_days=None)
del part; gc.collect()

columns = feature_columns(frame)
print(f"{len(frame):,} rows, {len(columns)} features")
print(f"'wins' in columns? {'wins' in columns}")
print(f"target-ish columns present in frame: {[c for c in frame.columns if 'target' in c or 'next' in c]}")
print(f"of those, leaking into features: {[c for c in columns if 'target' in c or 'next' in c]}\n")

y = frame["target_client_wins"].astype(bool)
for column in columns:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")

# Simple in-sample fit: a leak shows up instantly as near-perfect separation.
model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=0)
model.fit(frame[columns], y)
importance = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
print("top 15 features by importance:")
print(importance.head(15).to_string())

# Correlation of each feature with the label -- a leak is usually near +/-1.
print("\ntop 15 by |correlation with label|:")
correlations = frame[columns].corrwith(y.astype(float)).abs().sort_values(ascending=False)
print(correlations.head(15).to_string())

suspects = correlations[correlations > 0.9]
if not suspects.empty:
    print(f"\nLEAK CONFIRMED -- these features are >0.9 correlated with the label:\n{suspects.to_string()}")
else:
    print("\nNo single feature exceeds 0.9 correlation; leak may be a combination.")
