"""Active-day labelling + rank target + boosting-library comparison.

Best so far (calendar-window labels, 174 features): classification AUC 0.7435,
regression Spearman ~0 / sign 57%.

Tests three claims:
  1. Dropping the 7-day gap tolerance recovers infrequent accounts and does not
     hurt (or helps) AUC.
  2. A normalised rank target is learnable where a dollar target is not.
  3. LightGBM/XGBoost beat sklearn's HistGradientBoosting here.
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from trading_data.behaviour_features import build_active_day_frame, build_training_frame, feature_columns
from trading_data.bigquery_data_client import compact_memory
from trading_data.research import _rank_discrimination

PATH = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"
MIN_TRAIN, CADENCE = 20, 5


def load(builder, **kwargs):
    databases = sorted(pd.read_parquet(PATH, columns=["database"])["database"].unique())
    parts = []
    for database in databases:
        part = compact_memory(pd.read_parquet(PATH, filters=[("database", "==", database)]))
        parts.append(builder(part, **kwargs))
        del part; gc.collect()
    frame = pd.concat(parts, ignore_index=True)
    del parts; gc.collect()
    return frame


def make_model(name, classify):
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        return HistGradientBoostingClassifier(max_iter=150, random_state=0) if classify else HistGradientBoostingRegressor(max_iter=150, random_state=0)
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=0) if classify else lgb.LGBMRegressor(n_estimators=200, verbose=-1, random_state=0)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=200, tree_method="hist", verbosity=0, random_state=0) if classify else xgb.XGBRegressor(n_estimators=200, tree_method="hist", verbosity=0, random_state=0)
    raise ValueError(name)


def walk_forward(frame, target, columns, classify, model_name):
    days = sorted(frame["decision_day"].unique())
    model = make_model(model_name, classify)
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    fitted = False
    for offset in range(MIN_TRAIN, len(days)):
        test_mask = frame["decision_day"] == days[offset]
        if not test_mask.any():
            continue
        if (offset - MIN_TRAIN) % CADENCE == 0 or not fitted:
            train_mask = frame["decision_day"].isin(days[:offset])
            y = frame.loc[train_mask, target]
            valid = y.notna() & np.isfinite(y.astype(float))
            if valid.sum() > 100 and not (classify and y[valid].astype(bool).nunique() < 2):
                model.fit(frame.loc[train_mask, columns][valid.to_numpy()], y[valid])
                fitted = True
        if fitted:
            x = frame.loc[test_mask, columns]
            out.loc[test_mask] = model.predict_proba(x)[:, 1] if classify else model.predict(x)
    return out


print("Loading with ACTIVE-DAY labels (no gap tolerance)...", flush=True)
t0 = time.time()
active = load(build_active_day_frame, max_gap_days=None)
columns = feature_columns(active)
for column in columns:
    active[column] = pd.to_numeric(active[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
active["decision_day"] = pd.to_datetime(active["decision_day"]).dt.normalize()
print(f"  active-day rows: {len(active):,}  accounts: {active['account_key'].nunique():,}  "
      f"features: {len(columns)}  [{time.time()-t0:.0f}s]", flush=True)
print(f"  gap to next active day: median {active['days_until_next_active'].median():.0f}d, "
      f"p90 {active['days_until_next_active'].quantile(0.9):.0f}d, max {active['days_until_next_active'].max():.0f}d", flush=True)
print(f"  (calendar-window labelling kept only 386,847 rows -- see how many more survive here)\n", flush=True)

libraries = ["hgb"]
for name in ("lightgbm", "xgboost"):
    try:
        make_model(name, True); libraries.append(name)
    except Exception as exc:
        print(f"  {name} unavailable: {type(exc).__name__}", flush=True)

print("=== 1. CLASSIFICATION: P(client WINS next active day) -> A-book ===", flush=True)
for name in libraries:
    t0 = time.time()
    pred = walk_forward(active, "target_client_wins", columns, True, name)
    valid = pred.notna()
    disc = _rank_discrimination(pred[valid].to_numpy(), active.loc[valid, "target_client_wins"].astype(bool).to_numpy())
    print(f"  {name:<9} ROC AUC {disc['roc_auc']:.4f}  AP {disc['average_precision']:.4f}  "
          f"n={int(valid.sum()):,}  base={active.loc[valid,'target_client_wins'].mean():.3f}  [{time.time()-t0:.0f}s]", flush=True)

print("\n=== 2. RANK TARGET vs dollar target (both regression) ===", flush=True)
for target, label in (("target_rank", "rank [0,1]"), ("target_firm_value_usd", "dollars")):
    for name in libraries:
        t0 = time.time()
        pred = walk_forward(active, target, columns, False, name)
        valid = pred.notna() & active[target].notna()
        actual = active.loc[valid, target].astype(float)
        spearman = float(pd.Series(pred[valid].to_numpy()).rank().corr(actual.rank()))
        print(f"  {label:<12} {name:<9} Spearman {spearman:+.4f}  n={int(valid.sum()):,}  [{time.time()-t0:.0f}s]", flush=True)
