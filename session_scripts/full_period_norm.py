"""Target normalised by each account's FULL-PERIOD scale, measured against the
point-in-time version.

The argument for it: normalisation touches only the LABEL. Predictions still
come from legitimate features, and evaluation is in real dollars -- never in the
normalised units -- so full-period label engineering is defensible in a way a
feature leak never is. It also fixes a concrete problem: point-in-time scaling
needs 5 prior observations per account and covered only 70% of rows, discarding
exactly the newer accounts a routing desk most needs a view on.

The residual risk is narrow but real: a per-account scale computed over the
whole sample encodes that account's FUTURE volatility, and a model with enough
account-identifying features could partially infer it.

So this does not argue the point, it measures it. Both variants run side by
side; if they agree, the concern was immaterial. If full-period is dramatically
better, that gap IS the leak, quantified.
"""
import sys, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
MIN_TRAIN, CADENCE, TRAIL = 20, 10, 20
FIT = dict(n_estimators=100, num_leaves=31, subsample=0.5, subsample_freq=1,
           colsample_bytree=0.5, n_jobs=-1, verbose=-1, random_state=0)

frame = pd.read_parquet(f"{BASE}\\model_frame.parquet")
features = pd.read_csv(f"{BASE}\\model_features.csv", header=None)[0].tolist()
frame = frame.sort_values(["account_key", "decision_day"], kind="mergesort").reset_index(drop=True)
group = frame.groupby("account_key", observed=True)

frame["fwd_1"] = group["pnl"].shift(-1)
frame["fwd_5"] = group["pnl"].transform(
    lambda s: s[::-1].rolling(5, min_periods=1).sum()[::-1]).groupby(
    frame["account_key"], observed=True).shift(-1)

# A: point-in-time scale (trailing, shifted) -- the strict version.
frame["scale_pit"] = group["pnl"].transform(lambda s: s.rolling(TRAIL, min_periods=5).std().shift())
# B: full-period scale -- one constant per account over the whole sample.
frame["scale_full"] = group["pnl"].transform("std")
# C: full-period RANK of the forward outcome within the account's own history.
#    Purely ordinal, so account size cancels entirely.
for horizon in (1, 5):
    frame[f"rank_full_{horizon}"] = frame.groupby("account_key", observed=True)[f"fwd_{horizon}"].rank(pct=True)

for horizon in (1, 5):
    for tag, column in (("pit", "scale_pit"), ("full", "scale_full")):
        scale = frame[column].replace(0, np.nan) * np.sqrt(horizon)
        frame[f"sigma_{tag}_{horizon}"] = frame[f"fwd_{horizon}"] / scale

frame = frame.sort_values("decision_day", kind="mergesort").reset_index(drop=True)
X = np.ascontiguousarray(frame[features].to_numpy(dtype="float32"))
pnl = frame["pnl"].to_numpy(dtype="float64")
day_codes, day_values = pd.factorize(frame["decision_day"], sort=True)
starts = np.searchsorted(day_codes, np.arange(len(day_values) + 1))

TARGETS = {}
for horizon in (1, 5):
    TARGETS[f"A pit sigma {horizon}d>0"] = (frame[f"sigma_pit_{horizon}"].to_numpy(), horizon, 0.0)
    TARGETS[f"B full sigma {horizon}d>0"] = (frame[f"sigma_full_{horizon}"].to_numpy(), horizon, 0.0)
    TARGETS[f"C full rank {horizon}d>0.5"] = (frame[f"rank_full_{horizon}"].to_numpy(), horizon, 0.5)
    TARGETS[f"C full rank {horizon}d>0.9"] = (frame[f"rank_full_{horizon}"].to_numpy(), horizon, 0.9)
for name, (values, horizon, _) in TARGETS.items():
    print(f"  {name:<26} coverage {np.isfinite(values).mean():.0%}")
print(flush=True)


def walk_forward(values, threshold, horizon):
    score = np.full(len(X), np.nan)
    estimator, fitted = None, False
    for day in range(MIN_TRAIN, len(day_values)):
        lo, hi = starts[day], starts[day + 1]
        if hi <= lo:
            continue
        if (day - MIN_TRAIN) % CADENCE == 0 or not fitted:
            end = starts[max(0, day - horizon)]   # purge label overlap regardless of variant
            if end < 200:
                continue
            ok = np.isfinite(values[:end])
            if ok.sum() < 200:
                continue
            y = (values[:end] > threshold)[ok]
            if y.min() == y.max():
                continue
            estimator = lgb.LGBMClassifier(
                scale_pos_weight=float((~y).sum() / max(1, y.sum())), **FIT)
            estimator.fit(X[:end][ok], y)
            fitted = True
        if not fitted:
            continue
        score[lo:hi] = estimator.predict_proba(X[lo:hi])[:, 1]
    return score


def metrics(hedge_fraction, mask):
    kept = -pnl[mask] * (1.0 - hedge_fraction)
    daily = pd.Series(kept).groupby(day_codes[mask]).sum().sort_index()
    total = float(daily.sum())
    curve = daily.cumsum()
    dd = float((curve - curve.cummax()).min())
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() else 0.0
    return total, dd, sharpe, (total / abs(dd) if dd else np.inf)


t0 = time.time()
results = {name: walk_forward(v, t, h) for name, (v, h, t) in TARGETS.items()}
print(f"fitted {len(results)} models [{time.time()-t0:.0f}s]\n", flush=True)

reference = np.isfinite(next(iter(results.values())))
b_total, b_dd, b_sharpe, b_calmar = metrics(np.zeros(reference.sum()), reference)
print(f"{'strategy':<44}{'profit':>14}{'maxDD':>13}{'Sharpe':>8}{'Calmar':>8}")
print(f"{'FLAT B-BOOK':<44}${b_total:>13,.0f}${b_dd:>12,.0f}{b_sharpe:>8.2f}{b_calmar:>8.1f}")
for name, score in results.items():
    valid = np.isfinite(score)
    for pct in (0.02, 0.05, 0.10):
        s = score[valid]
        hedged = (s >= np.nanquantile(s, 1 - pct)).astype(float)
        total, dd, sharpe, calmar = metrics(hedged, valid)
        flag = "  <-- BEATS FLAT ON BOTH" if (total > b_total and dd > b_dd) else ""
        print(f"{name + f' | {int(pct*100)}%':<44}${total:>13,.0f}${dd:>12,.0f}"
              f"{sharpe:>8.2f}{calmar:>8.1f}{flag}", flush=True)

print("\nA vs B is the measurement: if full-period scaling beats point-in-time by a")
print("wide margin, that margin is the leak. If they agree, the concern was immaterial.")
