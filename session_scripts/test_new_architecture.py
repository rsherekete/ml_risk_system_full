"""Does the richer feature set + corrected labelling actually beat the baseline?

Baseline to beat, from the real 90-day BigQuery pull:
  classification ROC AUC 0.6505   regression Spearman -0.119 (anti-predictive)
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from trading_data.behaviour_features import build_training_frame, feature_columns
from trading_data.bigquery_data_client import compact_memory
from trading_data.research import _rank_discrimination

PATH = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"
NEEDED = ["account_key", "database", "platform", "region", "timestamp", "symbol", "cmd", "state",
          "volume_lots", "profit", "net_profit", "open_time", "close_time", "sl", "tp", "notional_usd"]

available = set(pd.read_parquet(PATH, columns=None).columns) if False else None
databases = sorted(pd.read_parquet(PATH, columns=["database"])["database"].unique())
parts = []
for database in databases:
    t0 = time.time()
    try:
        part = pd.read_parquet(PATH, columns=NEEDED, filters=[("database", "==", database)])
    except Exception:
        part = pd.read_parquet(PATH, filters=[("database", "==", database)])
    part = compact_memory(part)
    built = build_training_frame(part)
    parts.append(built)
    print(f"  {database:<18} {len(part):>10,} rows -> {len(built):>7,} labelled rows ({time.time()-t0:.0f}s)", flush=True)
    del part; gc.collect()

data = pd.concat(parts, ignore_index=True)
del parts; gc.collect()
columns = feature_columns(data)
print(f"\nlabelled rows: {len(data):,}  accounts: {data['account_key'].nunique():,}  features: {len(columns)}", flush=True)
print(f"features: {columns}\n", flush=True)

for column in columns:
    data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
data["decision_day"] = pd.to_datetime(data["decision_day"]).dt.normalize()

days = sorted(data["decision_day"].unique())
MIN_TRAIN, CADENCE = 20, 5


def walk_forward(frame, target, columns, kind, model_name):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classify = kind == "classification"
    if model_name == "gbm":
        model = HistGradientBoostingClassifier(max_iter=150, random_state=0) if classify else HistGradientBoostingRegressor(max_iter=150, random_state=0)
    else:
        estimator = LogisticRegression(max_iter=500, class_weight="balanced") if classify else Ridge(alpha=1.0)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)

    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    fitted = False
    for offset in range(MIN_TRAIN, len(days)):
        test_mask = frame["decision_day"] == days[offset]
        if not test_mask.any():
            continue
        if (offset - MIN_TRAIN) % CADENCE == 0 or not fitted:
            train_mask = frame["decision_day"].isin(days[:offset])
            y = frame.loc[train_mask, target]
            valid = y.notna()
            if valid.sum() > 50 and not (classify and y[valid].astype(bool).nunique() < 2):
                model.fit(frame.loc[train_mask, columns][valid.to_numpy()], y[valid])
                fitted = True
        if fitted:
            x_test = frame.loc[test_mask, columns]
            out.loc[test_mask] = model.predict_proba(x_test)[:, 1] if classify else model.predict(x_test)
    return out


print("=== CLASSIFICATION: will B-booking beat hedging tomorrow? ===", flush=True)
for model_name in ("linear", "gbm"):
    t0 = time.time()
    pred = walk_forward(data, "target_firm_gain", columns, "classification", model_name)
    valid = pred.notna() & data["target_firm_gain"].notna()
    disc = _rank_discrimination(pred[valid].to_numpy(), data.loc[valid, "target_firm_gain"].astype(bool).to_numpy())
    print(f"  {model_name:<7} ROC AUC {disc['roc_auc']:.4f}  AP {disc['average_precision']:.4f}  "
          f"n={int(valid.sum()):,}  base={data.loc[valid,'target_firm_gain'].mean():.3f}  [{time.time()-t0:.0f}s]", flush=True)

print("\n=== REGRESSION: predict the RATE, not the dollars ===", flush=True)
for target, label in (("target_firm_value_rate", "rate (new)"), ("target_firm_value_usd", "dollars (old way)")):
    for model_name in ("linear", "gbm"):
        t0 = time.time()
        pred = walk_forward(data, target, columns, "regression", model_name)
        valid = pred.notna() & data[target].notna() & np.isfinite(data[target])
        if valid.sum() < 100:
            print(f"  {label:<18} {model_name:<7} insufficient rows", flush=True)
            continue
        actual = data.loc[valid, target].astype(float)
        spearman = float(pd.Series(pred[valid].to_numpy()).rank().corr(actual.rank()))
        sign = float((np.sign(pred[valid]) == np.sign(actual)).mean())
        print(f"  {label:<18} {model_name:<7} Spearman {spearman:+.4f}  sign {sign:.3f}  n={int(valid.sum()):,}  [{time.time()-t0:.0f}s]", flush=True)
