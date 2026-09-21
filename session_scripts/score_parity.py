"""SCORE PARITY PROBE: is everything scoring ~0.8 live?
Scores a sample of TODAY'S closed trades through the parity-by-construction
builder + the deployed stance model, and compares the distribution to the
backtest artifact's (where only ~11.5% of trades pass the 0.80/0.20
anchors)."""
import sys, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from webapp import data_store
from webapp.trade_features import (build_features_for_scoring,
                                   SCORING_RAW_COLUMNS, TRADE_FEATURES)
from webapp import model_service

t0 = time.time()
# recent trades (last warehouse day) as the "live flow" sample
end = datetime.utcnow()
frame = data_store.read_history(start=end - timedelta(days=4), end=end)
frame["account_key"] = (frame["database"].astype(str) + ":"
                       + frame["login"].astype(str))
frame["open_time"] = pd.to_datetime(frame["open_time"])
last_day = frame["open_time"].dt.strftime("%Y-%m-%d").max()
todays = frame[frame["open_time"].dt.strftime("%Y-%m-%d") == last_day]
accounts = todays["account_key"].drop_duplicates().sample(
    n=min(80, todays["account_key"].nunique()), random_state=0)
print(f"[{time.time()-t0:.0f}s] sampling {len(accounts)} accounts' trades "
      f"from {last_day}", flush=True)

# the deployed stance model
import lightgbm as lgb
from pathlib import Path
ART = Path(r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts")
# EXACTLY the deployed stance model + its feature list
model = lgb.Booster(model_file=str(ART / "quant_model.txt"))
feature_names = (ART / "quant_model_features.txt").read_text(
    encoding="utf-8").splitlines()
print(f"deployed stance model loaded: {len(feature_names)} features",
      flush=True)

scores = []
hist_start = end - timedelta(days=120)
tape_all = data_store.read_history(start=hist_start, end=end)
tape_all["account_key"] = (tape_all["database"].astype(str) + ":"
                          + tape_all["login"].astype(str))
tape_all["open_time"] = pd.to_datetime(tape_all["open_time"])
for acc in accounts:
    sub = todays[todays["account_key"] == acc].sort_values("open_time")
    pick = sub.tail(3)
    tape = tape_all[(tape_all["account_key"] == acc)
                    & (~tape_all.index.isin(pick.index))]
    cols = [c for c in SCORING_RAW_COLUMNS if c in pick.columns]
    try:
        built = build_features_for_scoring(pick[cols].copy(),
                                           tape[cols].copy())
        X = built.reindex(columns=feature_names).astype(float).to_numpy()
        p = model.predict(X)
        scores.extend([float(v) for v in np.ravel(p)])
    except Exception as e:
        continue
scores = np.array(scores)
print(f"[{time.time()-t0:.0f}s] scored {len(scores)} live trades", flush=True)
if len(scores):
    q = np.percentile(scores, [5, 25, 50, 75, 95])
    print(f"LIVE distribution: p5 {q[0]:.3f} | p25 {q[1]:.3f} | "
          f"median {q[2]:.3f} | p75 {q[3]:.3f} | p95 {q[4]:.3f}")
    print(f"share >= 0.80: {np.mean(scores >= 0.8):.1%} | "
          f"share <= 0.20: {np.mean(scores <= 0.2):.1%} | "
          f"combined anchor-pass: {np.mean((scores >= 0.8) | (scores <= 0.2)):.1%}")
    print("BACKTEST reference (Aug OOS artifact): anchor-pass ~11.5%, "
          "median score ~0.5-0.6")
