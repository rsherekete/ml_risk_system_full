"""Judge every account against ITSELF, then re-weight by dollar impact.

The confound behind every failure so far: models rank accounts by SIZE, because
a big account's ordinary day dwarfs a small account's exceptional one. Ranking
by magnitude then hedges the big two-way payers whose losses are the firm's
revenue.

Orthogonalisation attacked that after the fact. This attacks it at the target:
express each account's forward P&L relative to its OWN typical performance, so
the model learns "is this unusually good FOR THIS ACCOUNT" -- a scale-free
question every account answers on equal terms.

Two normalisations, both strictly point-in-time (an account's own PRIOR history
only -- normalising by its full-period distribution would leak its future):
  * sigma units : forward P&L / trailing std of that account's daily P&L
  * self-rank   : percentile of forward P&L within that account's trailing window

Scale-free direction alone cannot drive routing, though -- the firm's P&L is in
dollars, and a confident call on a tiny account is worth nothing. So stage two
recombines:

    routing score = (scale-free direction) x (expected dollar magnitude)

which is exactly "prioritise by expected magnitude once direction is known".

All labels are purged by their horizon; the previous run showed overlapping
labels inflating results roughly 15x.
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
by_account = frame.groupby("account_key", observed=True)["pnl"]

# Forward windows (row's own day excluded -- it is the decision, not an outcome).
frame["fwd_1"] = by_account.shift(-1)
frame["fwd_5"] = by_account.transform(
    lambda s: s[::-1].rolling(5, min_periods=1).sum()[::-1]).groupby(
    frame["account_key"], observed=True).shift(-1)

# Point-in-time scale: trailing volatility of the account's OWN daily P&L,
# shifted so today contributes nothing to its own denominator.
frame["own_scale"] = by_account.transform(
    lambda s: s.rolling(TRAIL, min_periods=5).std().shift())
frame["own_mean"] = by_account.transform(
    lambda s: s.rolling(TRAIL, min_periods=5).mean().shift())

for horizon in (1, 5):
    scale = frame["own_scale"].replace(0, np.nan) * np.sqrt(horizon)
    # How many of its OWN typical daily moves is this, net of its own drift.
    frame[f"sigma_{horizon}"] = (frame[f"fwd_{horizon}"] - frame["own_mean"] * horizon) / scale

frame = frame.sort_values("decision_day", kind="mergesort").reset_index(drop=True)
X = np.ascontiguousarray(frame[features].to_numpy(dtype="float32"))
pnl = frame["pnl"].to_numpy(dtype="float64")
day_codes, day_values = pd.factorize(frame["decision_day"], sort=True)
starts = np.searchsorted(day_codes, np.arange(len(day_values) + 1))
sigma = {h: frame[f"sigma_{h}"].to_numpy(dtype="float64") for h in (1, 5)}
fwd = {h: frame[f"fwd_{h}"].to_numpy(dtype="float64") for h in (1, 5)}
print(f"X {X.shape} | sigma_1 coverage {np.isfinite(sigma[1]).mean():.0%}, "
      f"sigma_5 {np.isfinite(sigma[5]).mean():.0%}", flush=True)
print(f"sigma_1 spread: p10 {np.nanpercentile(sigma[1],10):+.2f}  "
      f"median {np.nanmedian(sigma[1]):+.2f}  p90 {np.nanpercentile(sigma[1],90):+.2f}\n", flush=True)


def walk_forward(target_fn, horizon, regression=False):
    score = np.full(len(X), np.nan)
    estimator, fitted = None, False
    for day in range(MIN_TRAIN, len(day_values)):
        lo, hi = starts[day], starts[day + 1]
        if hi <= lo:
            continue
        if (day - MIN_TRAIN) % CADENCE == 0 or not fitted:
            end = starts[max(0, day - horizon)]          # purge overlapping labels
            if end < 200:
                continue
            y = target_fn(end)
            if y is None:
                continue
            ok = np.isfinite(np.asarray(y, dtype="float64"))
            if ok.sum() < 200:
                continue
            if regression:
                estimator = lgb.LGBMRegressor(objective="huber", **FIT)
                estimator.fit(X[:end][ok], np.asarray(y)[ok])
            else:
                yb = np.asarray(y)[ok].astype(bool)
                if yb.min() == yb.max():
                    continue
                estimator = lgb.LGBMClassifier(
                    scale_pos_weight=float((~yb).sum() / max(1, yb.sum())), **FIT)
                estimator.fit(X[:end][ok], yb)
            fitted = True
        if not fitted:
            continue
        block = X[lo:hi]
        score[lo:hi] = (estimator.predict(block) if regression
                        else estimator.predict_proba(block)[:, 1])
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
scores = {}
for horizon in (1, 5):
    # Stage 1: scale-free direction -- an unusually GOOD day for this account.
    scores[f"self-sigma {horizon}d > 0"] = (walk_forward(
        lambda e, h=horizon: sigma[h][:e] > 0, horizon), horizon)
    # A strong positive outlier by the account's own standards.
    scores[f"self-sigma {horizon}d > 1"] = (walk_forward(
        lambda e, h=horizon: sigma[h][:e] > 1.0, horizon), horizon)
    scores[f"self-sigma {horizon}d regression"] = (walk_forward(
        lambda e, h=horizon: sigma[h][:e], horizon, regression=True), horizon)
# Stage 2 input: expected dollar magnitude, so a scale-free call can be priced.
magnitude = walk_forward(lambda e: np.abs(fwd[1][:e]), 1, regression=True)
print(f"fitted {len(scores)+1} models [{time.time()-t0:.0f}s]\n", flush=True)

reference = np.isfinite(next(iter(scores.values()))[0]) & np.isfinite(magnitude)
b_total, b_dd, b_sharpe, b_calmar = metrics(np.zeros(reference.sum()), reference)
print(f"{'strategy':<50}{'profit':>14}{'maxDD':>13}{'Sharpe':>8}{'Calmar':>8}")
print(f"{'FLAT B-BOOK':<50}${b_total:>13,.0f}${b_dd:>12,.0f}{b_sharpe:>8.2f}{b_calmar:>8.1f}")


def show(label, values, valid):
    for pct in (0.02, 0.05):
        s = values[valid]
        hedged = (s >= np.nanquantile(s, 1 - pct)).astype(float)
        total, dd, sharpe, calmar = metrics(hedged, valid)
        flag = "  <-- BEATS FLAT ON BOTH" if (total > b_total and dd > b_dd) else ""
        print(f"{label + f' | {int(pct*100)}%':<50}${total:>13,.0f}${dd:>12,.0f}"
              f"{sharpe:>8.2f}{calmar:>8.1f}{flag}", flush=True)


print("\n-- stage 1: scale-free direction alone --")
for label, (score, horizon) in scores.items():
    valid = np.isfinite(score) & np.isfinite(magnitude)
    show(label, score, valid)

print("\n-- stage 2: direction x expected dollar magnitude --")
for label, (score, horizon) in scores.items():
    valid = np.isfinite(score) & np.isfinite(magnitude)
    combined = np.full(len(X), np.nan)
    # Centre the direction score so a confident LOSS call becomes negative and
    # multiplying by dollars ranks it away from the hedge list rather than
    # toward it.
    centred = score - np.nanmedian(score[valid])
    combined[valid] = centred[valid] * np.clip(magnitude[valid], 0, None)
    show(f"{label} x $magnitude", combined, valid)
