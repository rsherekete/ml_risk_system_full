"""Dollar-weighted routing objectives, engineered to run in minutes not hours.

THE THREE THINGS THAT MADE THE PREVIOUS VERSION SLOW

1. Rebuilding the frame from 8 parquet shards on every run (~4 min, every
   script, every time). Now built once and cached to disk; subsequent runs load
   it in seconds.

2. `frame.loc[boolean_mask, columns]` inside the walk-forward loop. That copies
   the ENTIRE training matrix -- ~500k x 174 float32, assembled column by column
   out of a wide mixed-dtype frame -- on every refit, for every arm. This, not
   the boosting, was the real cost.

   The fix is structural rather than incremental: sort the rows by decision_day
   ONCE, so the training set for any day is a contiguous PREFIX of the matrix.
   `X[:k]` is then a zero-copy view instead of a 350MB materialisation, and
   LightGBM receives a contiguous float32 buffer it can bin directly.

3. Re-fitting from scratch at a 5-day cadence with 200 trees. An earlier sweep
   showed refit cadence is worth ~0.000x AUC, so 10 days and 100 subsampled
   trees cost essentially nothing and save an order of magnitude.

The objectives themselves are unchanged, and the leak guards stay in place.
"""
import gc, os, sys, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from trading_data.research import _rank_discrimination

BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
CACHE = f"{BASE}\\model_frame.parquet"
MIN_TRAIN, CADENCE = 20, 10
FIT = dict(n_estimators=100, num_leaves=31, subsample=0.5, subsample_freq=1,
           colsample_bytree=0.5, n_jobs=-1, verbose=-1, random_state=0)

t_start = time.time()
if not os.path.exists(CACHE):
    from trading_data.behaviour_features import build_active_day_frame, feature_columns
    from trading_data.bigquery_data_client import compact_memory
    # MEMORY. The previous version concatenated all 8 databases and THEN ran a
    # per-column `to_numeric(...).replace([inf,-inf], nan).astype(float32)` loop
    # across 174 columns. `.replace()` allocates several temporaries per column,
    # which drove resident memory to 12.8 GB against ~355 MB of actual data --
    # the process spent its time thrashing rather than fitting. (The same
    # allocate-temporaries trap already cost us once, in `compact_memory`.)
    #
    # Instead: reduce each part to exactly the columns needed and cast it to
    # float32 immediately, so peak memory is one part rather than eight, and do
    # the non-finite cleanup once in numpy where it is a single pass.
    features = None
    parts = []
    for database in sorted(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet",
                                           columns=["database"])["database"].unique()):
        part = compact_memory(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet",
                                              filters=[("database", "==", database)]))
        active = build_active_day_frame(part, max_gap_days=None)
        del part; gc.collect()
        if features is None:
            features = feature_columns(active)
        block = active[features].astype("float32", copy=False)
        block["decision_day"] = pd.to_datetime(active["decision_day"]).dt.normalize()
        block["pnl"] = pd.to_numeric(active["target_profit"], errors="coerce")
        block["account_key"] = active["account_key"].astype("string")
        parts.append(block.loc[block["pnl"].notna()])
        del active, block; gc.collect()
        print(f"  {database}: {len(parts[-1]):,} rows", flush=True)
    built = pd.concat(parts, ignore_index=True)
    del parts; gc.collect()
    # Single vectorised pass instead of 174 per-column replace() calls.
    matrix = built[features].to_numpy(dtype="float32")
    np.putmask(matrix, ~np.isfinite(matrix), np.nan)
    built[features] = matrix
    del matrix; gc.collect()
    # Sorting here is what makes every later training slice a contiguous view.
    built = built.sort_values("decision_day", kind="mergesort").reset_index(drop=True)
    keep = features + ["decision_day", "pnl", "account_key"]
    built[keep].to_parquet(CACHE, index=False)
    pd.Series(features).to_csv(f"{BASE}\\model_features.csv", index=False, header=False)
    print(f"built and cached frame in {time.time()-t_start:.0f}s", flush=True)
    del built; gc.collect()

frame = pd.read_parquet(CACHE)
features = pd.read_csv(f"{BASE}\\model_features.csv", header=None)[0].tolist()
LABELS = {"pnl", "label_wins", "abs_pnl", "win_size", "target_profit", "target_client_wins"}
leaked = sorted(set(features) & LABELS)
if leaked:
    raise RuntimeError(f"label columns present in feature set: {leaked}")

# One contiguous float32 matrix. Every training/test slice below is a view into
# this, so no per-refit copying happens at all.
X = np.ascontiguousarray(frame[features].to_numpy(dtype="float32"))
pnl = frame["pnl"].to_numpy(dtype="float64")
wins = pnl > 0
abs_pnl = np.abs(pnl)
day_codes, day_values = pd.factorize(frame["decision_day"], sort=True)
# Row boundaries per day; rows are sorted, so day d occupies [starts[d], starts[d+1]).
starts = np.searchsorted(day_codes, np.arange(len(day_values) + 1))
print(f"frame ready in {time.time()-t_start:.0f}s | X {X.shape} "
      f"{X.nbytes/1e9:.2f} GB | {len(day_values)} days", flush=True)
positive = pnl[wins]
print(f"win sizes: median ${np.median(positive):,.0f}, p99 ${np.quantile(positive, 0.99):,.0f}, "
      f"max ${positive.max():,.0f} | top 1% carry "
      f"{np.sort(positive)[-len(positive)//100:].sum()/positive.sum():.1%} of winner dollars\n", flush=True)


def walk_forward(kind, weight=None):
    score = np.full(len(X), np.nan)
    estimator, fitted = None, False
    t0 = time.time()
    for day in range(MIN_TRAIN, len(day_values)):
        lo, hi = starts[day], starts[day + 1]
        if hi <= lo:
            continue
        if (day - MIN_TRAIN) % CADENCE == 0 or not fitted:
            end = starts[day]
            if end < 200:
                continue
            X_train = X[:end]              # contiguous view -- no copy
            sample_weight = None
            if weight == "abs":
                sample_weight = np.clip(abs_pnl[:end], 1e-3, None)
            elif weight == "log":
                sample_weight = np.clip(np.log1p(abs_pnl[:end]), 1e-3, None)

            if kind == "clf":
                y = wins[:end]
                if y.min() == y.max():
                    continue
                estimator = lgb.LGBMClassifier(**FIT)
                estimator.fit(X_train, y, sample_weight=sample_weight)
            elif kind == "bigwin":
                train_positive = pnl[:end][wins[:end]]
                if len(train_positive) < 50:
                    continue
                y = pnl[:end] >= np.quantile(train_positive, 0.9)   # training-only threshold
                if y.min() == y.max():
                    continue
                estimator = lgb.LGBMClassifier(
                    scale_pos_weight=float((~y).sum() / max(1, y.sum())), **FIT)
                estimator.fit(X_train, y)
            elif kind == "huber":
                estimator = lgb.LGBMRegressor(objective="huber", **FIT)
                estimator.fit(X_train, pnl[:end], sample_weight=sample_weight)
            elif kind == "tweedie":
                estimator = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, **FIT)
                estimator.fit(X_train, np.clip(pnl[:end], 0, None))
            fitted = True
        if not fitted:
            continue
        block = X[lo:hi]
        score[lo:hi] = (estimator.predict_proba(block)[:, 1] if kind in {"clf", "bigwin"}
                        else estimator.predict(block))
    return score, time.time() - t0


baseline = None
print("=== objectives judged on FIRM DOLLARS ===")
print("(vs random > 0 is the bar: the RANKING beat hedging an arbitrary same-sized subset)\n")
for kind, weight, label in (
    ("clf", None, "A unweighted classifier"),
    ("clf", "abs", "B classifier x |P&L|"),
    ("clf", "log", "C classifier x log1p|P&L|"),
    ("bigwin", None, "D BIG-WIN classifier (top decile)"),
    ("huber", None, "E Huber regression on P&L"),
    ("tweedie", None, "F Tweedie regression on win size"),
):
    score, elapsed = walk_forward(kind, weight)
    valid = ~np.isnan(score)
    if not valid.any():
        print(f"  {label:<34} no scores", flush=True)
        continue
    s, p, w, a = score[valid], pnl[valid], wins[valid], abs_pnl[valid]
    baseline = -p.sum()
    auc = _rank_discrimination(s, w)["roc_auc"]
    if auc > 0.95:
        raise RuntimeError(f"{label}: AUC {auc:.4f} indicates leakage, not skill")
    order = pd.Series(s).rank(pct=True).to_numpy()
    dollar_auc = float((order * a)[w].sum() / max(1e-9, a[w].sum()))
    print(f"  {label:<34} AUC {auc:.4f} | dollar-AUC {dollar_auc:.4f} [{elapsed:.0f}s]")
    for pct in (0.05, 0.10, 0.20):
        hedged = s >= np.quantile(s, 1 - pct)
        firm = float(-p[~hedged].sum())
        captured = p[hedged & w].sum() / max(1e-9, p[w].sum())
        print(f"      hedge {int(pct*100):>2}%: firm ${firm:>13,.0f}  vs flat {firm-baseline:>+13,.0f}  "
              f"vs random {firm-baseline*(1-pct):>+13,.0f}  winner-$ captured {captured:>5.1%}", flush=True)

print(f"\nflat B-book: ${baseline:,.0f}   (total elapsed {time.time()-t_start:.0f}s)")
