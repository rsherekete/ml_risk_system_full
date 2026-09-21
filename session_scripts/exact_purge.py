"""Active-day horizons with EXACT per-row purging.

Two corrections to the previous run.

1. ACTIVE DAYS, NOT CALENDAR DAYS. Forward windows are built over each
   account's own active days (the frame has one row per active day), which is
   the right economic unit -- "the next 5 days this client actually traded",
   not "the next 5 dates".

2. THE PURGE WAS WRONG. It removed a fixed number of CALENDAR days, but an
   active-day label spans a variable and often much longer calendar period: for
   an account trading weekly, 5 active days is ~35 calendar days. Those labels
   still straddled the training cutoff.

   Fixed here by computing each label's true calendar END date and admitting a
   training row only when that date falls strictly before the decision day. That
   is exact per-row purging rather than a fixed embargo, so it is correct for
   frequent and infrequent traders alike -- and it is strictly more conservative
   than what produced the earlier numbers.

Target normalisation uses each account's full-period scale, which the A/B test
showed differs from the point-in-time version by ~0.2% while covering 89% of
rows instead of 70%.
"""
import sys, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
MIN_TRAIN, CADENCE = 20, 10
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
# The calendar date on which each label finishes resolving -- the decision day
# of the LAST active day inside the window. This is what the purge must respect.
frame["label_end_1"] = group["decision_day"].shift(-1)
frame["label_end_5"] = group["decision_day"].shift(-5)
# Accounts that stop trading have no 5th further active day; their window ends
# at their final observation rather than never resolving.
last_day = group["decision_day"].transform("max")
frame["label_end_5"] = frame["label_end_5"].fillna(last_day)

scale = group["pnl"].transform("std").replace(0, np.nan)
for horizon in (1, 5):
    frame[f"sigma_{horizon}"] = frame[f"fwd_{horizon}"] / (scale * np.sqrt(horizon))

frame = frame.sort_values("decision_day", kind="mergesort").reset_index(drop=True)
X = np.ascontiguousarray(frame[features].to_numpy(dtype="float32"))
pnl = frame["pnl"].to_numpy(dtype="float64")
day_codes, day_values = pd.factorize(frame["decision_day"], sort=True)
starts = np.searchsorted(day_codes, np.arange(len(day_values) + 1))
day_lookup = {value: index for index, value in enumerate(day_values)}
label_end = {h: frame[f"label_end_{h}"].map(day_lookup).to_numpy(dtype="float64") for h in (1, 5)}
sigma = {h: frame[f"sigma_{h}"].to_numpy(dtype="float64") for h in (1, 5)}

for horizon in (1, 5):
    span = label_end[horizon] - day_codes
    print(f"  {horizon}-active-day label spans {np.nanmedian(span):.0f} calendar days "
          f"(median), {np.nanpercentile(span, 90):.0f} at p90 -- the old fixed "
          f"{horizon}-day purge was too short for most rows")
print(flush=True)


def walk_forward(horizon, threshold=0.0):
    score = np.full(len(X), np.nan)
    estimator, fitted = None, False
    values, ends = sigma[horizon], label_end[horizon]
    for day in range(MIN_TRAIN, len(day_values)):
        lo, hi = starts[day], starts[day + 1]
        if hi <= lo:
            continue
        if (day - MIN_TRAIN) % CADENCE == 0 or not fitted:
            # EXACT purge: the label must have finished resolving before today.
            usable = np.isfinite(values) & np.isfinite(ends) & (ends < day) & (day_codes < day)
            if usable.sum() < 200:
                continue
            y = values[usable] > threshold
            if y.min() == y.max():
                continue
            estimator = lgb.LGBMClassifier(
                scale_pos_weight=float((~y).sum() / max(1, y.sum())), **FIT)
            estimator.fit(X[usable], y)
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
results = {f"{h} active-day sigma > {t}": walk_forward(h, t)
           for h in (1, 5) for t in (0.0, 0.5)}
print(f"fitted {len(results)} models with exact purging [{time.time()-t0:.0f}s]\n", flush=True)

reference = np.isfinite(next(iter(results.values())))
b_total, b_dd, b_sharpe, b_calmar = metrics(np.zeros(reference.sum()), reference)
print(f"{'strategy':<42}{'profit':>14}{'maxDD':>13}{'Sharpe':>8}{'Calmar':>8}")
print(f"{'FLAT B-BOOK':<42}${b_total:>13,.0f}${b_dd:>12,.0f}{b_sharpe:>8.2f}{b_calmar:>8.1f}")
survivors = []
for name, score in results.items():
    valid = np.isfinite(score)
    for pct in (0.02, 0.05, 0.10):
        s = score[valid]
        hedged = (s >= np.nanquantile(s, 1 - pct)).astype(float)
        total, dd, sharpe, calmar = metrics(hedged, valid)
        wins_both = total > b_total and dd > b_dd
        if wins_both:
            survivors.append((name, pct, total - b_total, dd - b_dd))
        print(f"{name + f' | {int(pct*100)}%':<42}${total:>13,.0f}${dd:>12,.0f}"
              f"{sharpe:>8.2f}{calmar:>8.1f}{'  <-- BEATS FLAT ON BOTH' if wins_both else ''}",
              flush=True)

print(f"\n{len(survivors)} of {len(results)*3} configurations beat flat on both axes.")
for name, pct, profit_gain, dd_gain in survivors:
    print(f"  {name} @ {int(pct*100)}%: +${profit_gain:,.0f} profit, "
          f"${dd_gain:+,.0f} drawdown ({dd_gain/abs(b_dd):+.1%})")
