"""Do execution-quality features beat the 0.7622 behavioural baseline?

Also: how does the model's ranking compare with the desk's ACTUAL book
decisions and the firm's own account_is_toxic label -- neither of which this
system has ever been measured against.
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from google.cloud import bigquery
from trading_data.behaviour_features import build_active_day_frame, feature_columns
from trading_data.bigquery_data_client import compact_memory
from trading_data.execution_quality import EVALUATION_ONLY_COLUMNS, attach_execution_features, fetch_execution_features
from trading_data.research import _rank_discrimination

PATH = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"
START, END = "2026-05-29", "2026-08-28"
MIN_TRAIN, CADENCE = 20, 5

print("building behavioural features...", flush=True)
parts = []
for database in sorted(pd.read_parquet(PATH, columns=["database"])["database"].unique()):
    part = compact_memory(pd.read_parquet(PATH, filters=[("database", "==", database)]))
    parts.append(build_active_day_frame(part, max_gap_days=None))
    del part; gc.collect()
base = pd.concat(parts, ignore_index=True)
del parts; gc.collect()
base["day"] = pd.to_datetime(base["day"])
print(f"  {len(base):,} rows, {base['account_key'].nunique():,} accounts", flush=True)

print("fetching execution quality from slippage_monitoring...", flush=True)
client = bigquery.Client(project="zfx-dwh-prod")
t0 = time.time()
execution = fetch_execution_features(client, START, END)
print(f"  {len(execution):,} account-days, {execution['account_key'].nunique():,} accounts [{time.time()-t0:.0f}s]", flush=True)

enriched = attach_execution_features(base, execution, include_evaluation_columns=True)
coverage = enriched["execution_events"].notna().mean()
print(f"  coverage: {coverage:.1%} of behavioural rows have execution context\n", flush=True)

feature_sets = {
    "behavioural only": [c for c in feature_columns(base)],
    "behavioural + execution": [c for c in feature_columns(enriched) if c not in EVALUATION_ONLY_COLUMNS],
}

for name, columns in feature_sets.items():
    frame = enriched.copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    frame["decision_day"] = pd.to_datetime(frame["decision_day"]).dt.normalize()
    days = sorted(frame["decision_day"].unique())

    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)
    pred = pd.Series(np.nan, index=frame.index, dtype="float64")
    fitted = False
    t0 = time.time()
    for offset in range(MIN_TRAIN, len(days)):
        test_mask = frame["decision_day"] == days[offset]
        if not test_mask.any():
            continue
        if (offset - MIN_TRAIN) % CADENCE == 0 or not fitted:
            train_mask = frame["decision_day"].isin(days[:offset])
            y = frame.loc[train_mask, "target_client_wins"]
            if y.notna().sum() > 100 and y.astype(bool).nunique() >= 2:
                model.fit(frame.loc[train_mask, columns], y.astype(bool))
                fitted = True
        if fitted:
            pred.loc[test_mask] = model.predict_proba(frame.loc[test_mask, columns])[:, 1]
    valid = pred.notna()
    disc = _rank_discrimination(pred[valid].to_numpy(), frame.loc[valid, "target_client_wins"].astype(bool).to_numpy())
    print(f"{name:<26} ROC AUC {disc['roc_auc']:.4f}  AP {disc['average_precision']:.4f}  "
          f"n={int(valid.sum()):,}  features={len(columns)}  [{time.time()-t0:.0f}s]", flush=True)
    enriched[f"pred_{name.replace(' ', '_').replace('+','plus')}"] = pred

print("\n=== Validation against the firm's OWN labels (never used as features) ===", flush=True)
scored = enriched.loc[enriched["pred_behavioural_plus_execution"].notna()].copy()

if "account_is_toxic" in scored.columns:
    toxic = scored["account_is_toxic"].astype("string").str.lower().eq("yes")
    if toxic.nunique() > 1:
        # Does a model trained purely to predict "client wins tomorrow" also
        # separate the accounts the firm independently flagged as toxic?
        disc = _rank_discrimination((-scored.loc[toxic.notna(), "pred_behavioural_plus_execution"]).to_numpy(),
                                    toxic[toxic.notna()].to_numpy())
        print(f"  P(client loses) vs firm's account_is_toxic: ROC AUC {disc['roc_auc']:.4f}  "
              f"(toxic rate {toxic.mean():.1%}, n={int(toxic.notna().sum()):,})", flush=True)

if "book" in scored.columns:
    actual_a = scored["book"].astype("string").str.upper().str.startswith("A")
    print(f"  desk's actual routing in window: A-book {actual_a.mean():.1%}, B-book {(~actual_a).mean():.1%}")
    disc = _rank_discrimination(scored["pred_behavioural_plus_execution"].to_numpy(), actual_a.fillna(False).to_numpy())
    print(f"  P(client wins) vs desk's actual A-book choice: ROC AUC {disc['roc_auc']:.4f}", flush=True)
    # Who was actually right? Compare realised outcomes under each.
    wins = scored["target_client_wins"].astype(bool)
    print(f"  clients the desk A-booked who then WON:   {wins[actual_a.fillna(False)].mean():.1%}")
    print(f"  clients the desk B-booked who then WON:   {wins[~actual_a.fillna(True)].mean():.1%}")
    print("  (A-book should have the HIGHER win rate -- those are the ones worth hedging away)", flush=True)
