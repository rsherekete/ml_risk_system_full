"""Regime + refit-cadence + server-as-feature comparison on real 90-day BQ data.

Refitting every 5 days instead of daily cuts ~70 fits per regime to ~14, which
is what makes a sweep this wide affordable. A model fitted from data strictly
before day D and reused for D..D+4 still never sees the future.
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.research import (
    TrainingWindow, compare_training_windows, daily_account_features, supervised_dataset,
)
from trading_data.bigquery_data_client import compact_memory

PATH = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"
NEEDED = ["account_key", "database", "platform", "region", "timestamp", "symbol",
          "cmd", "state", "volume_lots", "profit", "net_profit"]

databases = sorted(pd.read_parquet(PATH, columns=["database"])["database"].unique())
frames = []
for database in databases:
    t0 = time.time()
    part = compact_memory(pd.read_parquet(PATH, columns=NEEDED, filters=[("database", "==", database)]))
    features = daily_account_features(part)
    frames.append(features)
    print(f"  {database:<18} {len(part):>10,} rows -> {len(features):>7,} feature rows ({time.time()-t0:.0f}s)", flush=True)
    del part
    gc.collect()

features = pd.concat(frames, ignore_index=True)
del frames; gc.collect()
dataset = supervised_dataset(features)
del features; gc.collect()

exclude = {"account_key", "database", "platform", "decision_day", "target_profit", "target_loss"}
for column in [c for c in dataset.columns if c not in exclude]:
    dataset[column] = pd.to_numeric(dataset[column], errors="coerce").astype("float32")
dataset["target_profit"] = pd.to_numeric(dataset["target_profit"], errors="coerce").astype("float64")
dataset["target_loss"] = dataset["target_loss"].astype(bool)
days = pd.to_datetime(dataset["decision_day"]).dt.normalize().nunique()
print(f"\ndataset: {len(dataset):,} rows, {dataset['account_key'].nunique():,} accounts, {days} days\n", flush=True)

REGIMES = [
    TrainingWindow(mode="expanding", refit_every_days=5),
    TrainingWindow(mode="rolling", window_days=40, refit_every_days=5),
    TrainingWindow(mode="decay", half_life_days=20.0, refit_every_days=5),
    TrainingWindow(mode="expanding", warm_start=True),
]

print("=== CLASSIFICATION (ranks losing account-days) ===", flush=True)
t0 = time.time()
print(compare_training_windows(dataset, windows=REGIMES, task="classification").to_string(index=False), flush=True)
print(f"[{time.time()-t0:.0f}s]\n", flush=True)

print("=== REGRESSION (drives the risk-budget router) ===", flush=True)
t0 = time.time()
print(compare_training_windows(dataset, windows=REGIMES, task="regression").to_string(index=False), flush=True)
print(f"[{time.time()-t0:.0f}s]\n", flush=True)

# Does knowing WHICH server a client is on carry signal? The pooled model
# currently cannot tell an mt4_live01 client from an mt5_live01 one at all.
print("=== SERVER-AS-FEATURE: does per-server identity add signal? ===", flush=True)
with_server = dataset.copy()
for database in sorted(with_server["database"].unique()):
    with_server[f"is_{database}"] = (with_server["database"] == database).astype("float32")
best = TrainingWindow(mode="decay", half_life_days=20.0, refit_every_days=5)
t0 = time.time()
baseline = compare_training_windows(dataset, windows=[best], task="classification")
enriched = compare_training_windows(with_server, windows=[best], task="classification")
print(f"  without server features: ROC AUC {baseline.iloc[0]['roc_auc']:.4f}", flush=True)
print(f"  with    server features: ROC AUC {enriched.iloc[0]['roc_auc']:.4f}", flush=True)
print(f"[{time.time()-t0:.0f}s]", flush=True)
