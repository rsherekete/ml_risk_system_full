"""EUREKA BENCHMARK: is behavioral anomaly a tradeable signal?

Model the 'normal crowd' in behavioral-feature space (unsupervised, no
outcomes). Score each real trade's anomaly. Then test three things:
  1. Does win rate vary across the anomaly spectrum? (is anomaly informative)
  2. Copy-P&L for anomalous vs normal trades (with/against the crowd)
  3. Does anomaly ADD AUC to the base win-predictor? (new info, or redundant)
Small sample, fast, honest -- outcomes never touch the anomaly model.
"""
import sys
import time

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from webapp.trade_features import TRADE_FEATURES  # noqa: E402

frame = pd.read_parquet(ms.SCRATCH / "quant_feature_cache.parquet")
frame = frame.sort_values("open_time").reset_index(drop=True)
frame = frame.sample(n=min(300_000, len(frame)), random_state=0) \
    .sort_values("open_time").reset_index(drop=True)
print(f"sample: {len(frame):,} real trades")

# BEHAVIORAL features only -- how the trade was PLACED in context. No
# history-outcome aggregates, so anomaly = "unusual action", not "unusual
# past P&L".
behaviour = ["direction", "log_lots", "log_notional", "hour", "weekday",
             "sl_distance", "tp_distance", "risk_reward", "has_sl", "has_tp",
             "ctx_return_1h", "ctx_return_4h", "ctx_vol_24h", "zscore_24h",
             "range_pos_24h", "with_momentum", "momentum_align",
             "notional_vs_usual", "trades_today", "hours_since_last"]
behaviour = [b for b in behaviour if b in frame.columns]
Xb = frame[behaviour].to_numpy(dtype="float64")
Xb = np.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)

t0 = time.time()
forest = IsolationForest(n_estimators=150, max_samples=50_000,
                         contamination="auto", n_jobs=-1, random_state=0)
forest.fit(Xb)
anomaly = -forest.score_samples(Xb)          # higher = more anomalous
frame["anomaly"] = anomaly
print(f"anomaly model fit in {time.time()-t0:.0f}s "
      f"(range {anomaly.min():.2f}-{anomaly.max():.2f})")

pnl = pd.to_numeric(frame["net_profit"], errors="coerce").fillna(0).to_numpy()
win = pnl > 0

# 1. win rate + edge across the anomaly spectrum
print("\n1) OUTCOME ACROSS THE ANOMALY SPECTRUM (deciles, most-normal -> most-anomalous):")
frame["_bucket"] = pd.qcut(anomaly, 10, labels=False, duplicates="drop")
for b in sorted(frame["_bucket"].dropna().unique()):
    mask = (frame["_bucket"] == b).to_numpy()
    print(f"  decile {int(b)}: win {win[mask].mean():.1%} | "
          f"mean P&L ${pnl[mask].mean():+7.2f} | n={mask.sum():,}")

# 2. informed (top-decile anomaly) vs crowd (bottom-decile): copy edge
top = anomaly >= np.quantile(anomaly, 0.9)
bot = anomaly <= np.quantile(anomaly, 0.1)
print("\n2) COPY EDGE by group:")
print(f"  informed (anomalous top 10%): copy P&L ${pnl[top].mean():+.2f}/trade "
      f"| win {win[top].mean():.1%}")
print(f"  crowd (normal bottom 10%)   : copy P&L ${pnl[bot].mean():+.2f}/trade "
      f"| win {win[bot].mean():.1%}")

# 3. does anomaly ADD AUC beyond the full model? the decisive benchmark
split = int(len(frame) * 0.7)
X_full = np.nan_to_num(frame[TRADE_FEATURES].to_numpy(dtype="float32"),
                       nan=np.nan)
np.putmask(X_full, ~np.isfinite(X_full), np.nan)
y = win.astype(int)


def auc_of(cols):
    m = lgb.LGBMClassifier(n_estimators=150, num_leaves=63, learning_rate=0.08,
                           max_bin=63, n_jobs=-1, verbose=-1, random_state=0)
    m.fit(cols[:split], y[:split])
    return roc_auc_score(y[split:], m.predict_proba(cols[split:])[:, 1])

base = auc_of(X_full)
augmented = auc_of(np.column_stack([X_full, anomaly]))
anomaly_only = auc_of(anomaly.reshape(-1, 1))
print("\n3) DOES ANOMALY ADD SIGNAL? (decisive benchmark)")
print(f"  anomaly alone      : AUC {anomaly_only:.4f}")
print(f"  full model         : AUC {base:.4f}")
print(f"  full model + anomaly: AUC {augmented:.4f}  (delta {augmented-base:+.4f})")
print("\nverdict: " + (
    "ANOMALY ADDS SIGNAL -- informed-flow filter is real"
    if augmented - base > 0.003 else
    "anomaly is redundant with existing features (already captured)"))
print("DONE")
