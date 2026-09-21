"""S2 class x horizon grid: per-CLASS models (fx / index / crypto) at 15m
and 1h, each with in-class symbol weights. Decides whether per-class models
beat the pooled one and which horizon each class wants."""
import sys

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from webapp.trade_features import (  # noqa: E402
    symbol_class, trade_cost_vector)

S2_FEATURES = [
    "ctx_return_1h", "ctx_return_4h", "ctx_return_24h", "ctx_vol_24h",
    "ctx_flow_prev", "zscore_24h", "range_pos_24h", "vol_regime",
    "mom_1h_vol", "mom_4h_vol", "mom_24h_vol", "with_momentum",
    "momentum_align", "hour", "weekday", "direction", "log_lots",
    "log_notional", "symbol_code", "trades_today", "hours_since_last",
]

frame = pd.read_parquet(ms.SCRATCH / "quant_feature_cache.parquet")
frame["s2_class"] = frame["symbol"].astype(str).map(symbol_class)
frame = frame.loc[frame["s2_class"].isin(["fx", "index/other", "crypto"])]
frame = frame.sort_values("open_time").reset_index(drop=True)
n = len(frame)
print(f"non-gold universe: {n:,} trades")

bars = pd.read_parquet(ms.SCRATCH / "exit_study_bars.parquet")
bars["minute"] = pd.to_datetime(bars["minute"])
index = {s: (g.sort_values("minute")["minute"].to_numpy(),
             g.sort_values("minute")["close"].to_numpy())
         for s, g in bars.groupby("symbol", observed=True)}

open_np = pd.to_datetime(frame["open_time"]).to_numpy()
entry = pd.to_numeric(frame["open_price"], errors="coerce").to_numpy()
direction = pd.to_numeric(frame["direction"], errors="coerce").to_numpy()


def horizon_price(seconds):
    out = np.full(n, np.nan)
    for symbol, group_index in frame.groupby("symbol", observed=True).groups.items():
        if symbol not in index:
            continue
        minutes, closes = index[symbol]
        rows = np.fromiter(group_index, dtype=int)
        target = open_np[rows] + np.timedelta64(seconds, "s")
        position = np.searchsorted(minutes, target, side="right") - 1
        valid = position >= 0
        gap = (target - minutes[np.clip(position, 0, None)]) / np.timedelta64(1, "s")
        out[rows] = np.where(valid & (gap <= 600),
                             closes[np.clip(position, 0, None)], np.nan)
    return out


def per_unit(symbol):
    return {"crypto": 1.0, "gold": 100.0, "silver": 100.0,
            "fx": 100_000.0}.get(symbol_class(symbol), 100.0)


pu = frame["symbol"].map(per_unit).to_numpy(dtype="float64")
cost_lot = trade_cost_vector(frame["symbol"], frame["open_price"],
                             pd.Series(1.0, index=frame.index))
X_all = np.ascontiguousarray(frame[S2_FEATURES].to_numpy(dtype="float32"))
np.putmask(X_all, ~np.isfinite(X_all), np.nan)
symbols_all = frame["symbol"].astype(str).to_numpy()
classes_all = frame["s2_class"].to_numpy()

print(f"\n{'class':<14}{'horizon':>8}{'rows':>10}{'AUC':>8}"
      f"{'fade decile $/lot':>19}{'trades/day':>12}")
for seconds, horizon_name in ((300, "5m"), (900, "15m")):
    p_h = horizon_price(seconds)
    profit = (p_h - entry) * direction * pu - cost_lot
    for cls in ("fx", "index/other", "crypto", "POOLED"):
        mask = ((classes_all == cls) if cls != "POOLED"
                else np.ones(n, dtype=bool))
        mask = mask & np.isfinite(profit) & (entry > 0)
        rows = np.flatnonzero(mask)
        if len(rows) < 15_000:
            print(f"{cls:<14}{horizon_name:>8}{len(rows):>10,}   too thin")
            continue
        y = (profit[rows] > 0).astype(int)
        counts = pd.Series(symbols_all[rows]).value_counts()
        w = 1.0 / np.sqrt(pd.Series(symbols_all[rows]).map(counts)
                          .to_numpy(dtype="float64"))
        w *= len(w) / w.sum()
        split = int(len(rows) * 0.8)
        if len(set(y[:split])) < 2 or len(set(y[split:])) < 2:
            continue
        model = lgb.LGBMClassifier(n_estimators=200, num_leaves=31,
                                   learning_rate=0.08, max_bin=63, n_jobs=-1,
                                   verbose=-1, random_state=0)
        model.fit(X_all[rows[:split]], y[:split], sample_weight=w[:split])
        scores = model.predict_proba(X_all[rows[split:]])[:, 1]
        auc = roc_auc_score(y[split:], scores)
        lo = np.quantile(scores, 0.10)
        fade = -profit[rows[split:]][scores <= lo]
        days = pd.to_datetime(frame["open_time"]).iloc[rows[split:]] \
            .dt.normalize().nunique()
        print(f"{cls:<14}{horizon_name:>8}{len(rows):>10,}{auc:>8.4f}"
              f"{fade.mean():>+19.2f}{len(fade) / max(1, days):>12,.0f}")

print("\nDONE")
