"""Realign the objective with the economics: dollar-weighted targets.

Every model so far optimised a loss in which a $50 account-day and a $50,000
account-day count the same. The firm's P&L does not work that way, so the
objective has been misaligned with the decision from the beginning -- a model
can score 0.76 AUC by being right about thousands of trivial rows while missing
the handful that carry the money.

Six restructured objectives, all judged on FIRM DOLLARS rather than row accuracy:

  A  unweighted classifier                 (reference: what we have now)
  B  classifier weighted by |P&L|          (each row counts its dollars)
  C  classifier weighted by log1p(|P&L|)   (tempered -- pure |P&L| lets one
                                            outlier dominate a whole refit)
  D  BIG-WIN classifier                    (target = winner in the top decile of
                                            win sizes; directly the rare event
                                            that actually costs the firm)
  E  Huber regression on signed P&L        (total subsequent profit, robust)
  F  Tweedie regression on win size        (heavy-tailed, non-negative -- the
                                            right family for this shape)

Judged by dollar-weighted AUC, the share of winner-dollars captured, and firm
P&L against BOTH flat B-book and a random hedge of identical size.
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from trading_data.behaviour_features import build_active_day_frame, feature_columns
from trading_data.bigquery_data_client import compact_memory
from trading_data.research import _rank_discrimination

BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
MIN_TRAIN, CADENCE = 20, 10

# Cheap-but-faithful fit settings. The first version of this script reused the
# timings from a LEAKED run, where every tree converged on one perfect split and
# an arm finished in 101s; honest fitting on 500k rows x 174 features is roughly
# 30x that. Halving the trees, subsampling rows/features and refitting every 10
# days instead of 5 costs very little discrimination (earlier sweeps showed
# refit cadence worth ~0.000x AUC) and turns hours into minutes.
FIT = dict(n_estimators=100, num_leaves=31, subsample=0.5, subsample_freq=1,
           colsample_bytree=0.5, n_jobs=-1, verbose=-1, random_state=0)

parts = []
for database in sorted(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", columns=["database"])["database"].unique()):
    part = compact_memory(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", filters=[("database", "==", database)]))
    parts.append(build_active_day_frame(part, max_gap_days=None))
    del part; gc.collect()
frame = pd.concat(parts, ignore_index=True)
del parts; gc.collect()

columns = feature_columns(frame)
for column in columns:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
frame["decision_day"] = pd.to_datetime(frame["decision_day"]).dt.normalize()
frame["pnl"] = pd.to_numeric(frame["target_profit"], errors="coerce")
frame = frame.loc[frame["pnl"].notna()].reset_index(drop=True)
frame["label_wins"] = frame["pnl"] > 0
frame["abs_pnl"] = frame["pnl"].abs()
frame["win_size"] = frame["pnl"].clip(lower=0)

# The rare event that drives hedging economics: a winner in the top decile of
# win sizes. Threshold from TRAINING data only inside the loop -- a global
# quantile here would be exactly the lookahead that inflated an earlier result.
# LEAK GUARD. An earlier version of this script assigned the target to
# `frame["wins"]` -- but `wins` is a LEGITIMATE feature (did the account win
# TODAY), so that line silently replaced a feature with tomorrow's answer and
# produced ROC AUC 1.0000 and a fictitious +$45M. Label columns are now
# `label_`-prefixed and their absence from the feature set is asserted, because
# a convincing wrong number is far more dangerous here than a crash.
LABEL_COLUMNS = {"label_wins", "pnl", "abs_pnl", "win_size", "target_profit", "target_client_wins"}
leaked = sorted(set(columns) & LABEL_COLUMNS)
if leaked:
    raise RuntimeError(f"label columns present in feature set: {leaked}")

days = sorted(frame["decision_day"].unique())
print(f"{len(frame):,} account-days, {frame['account_key'].nunique():,} accounts, {len(days)} days")
positive = frame.loc[frame["label_wins"], "pnl"]
print(f"win sizes: median ${positive.median():,.0f}, p90 ${positive.quantile(0.9):,.0f}, "
      f"p99 ${positive.quantile(0.99):,.0f}, max ${positive.max():,.0f}")
print(f"top 1% of winners carry {positive.nlargest(max(1, len(positive)//100)).sum()/positive.sum():.1%} "
      f"of ALL winner dollars\n", flush=True)


def walk_forward(kind, weight=None, label=""):
    """One arm, walk-forward. `kind` picks the objective; `weight` the emphasis."""
    score = pd.Series(np.nan, index=frame.index, dtype="float64")
    fitted = False
    estimator = None
    t0 = time.time()
    for offset in range(MIN_TRAIN, len(days)):
        test_mask = frame["decision_day"] == days[offset]
        if not test_mask.any():
            continue
        if (offset - MIN_TRAIN) % CADENCE == 0 or not fitted:
            train = frame["decision_day"].isin(days[:offset])
            X = frame.loc[train, columns]
            if len(X) < 200:
                continue
            sample_weight = None
            if weight == "abs":
                sample_weight = frame.loc[train, "abs_pnl"].to_numpy()
            elif weight == "log":
                sample_weight = np.log1p(frame.loc[train, "abs_pnl"].to_numpy())
            if sample_weight is not None:
                # A weight vector of all zeros (or one huge spike) makes the fit
                # degenerate; floor it so every row retains some influence.
                sample_weight = np.clip(sample_weight, 1e-3, None)

            if kind == "clf":
                y = frame.loc[train, "label_wins"]
                if y.nunique() < 2:
                    continue
                estimator = lgb.LGBMClassifier(**FIT)
                estimator.fit(X, y, sample_weight=sample_weight)
            elif kind == "bigwin":
                # Threshold computed on TRAINING rows only.
                train_positive = frame.loc[train & frame["label_wins"], "pnl"]
                if len(train_positive) < 50:
                    continue
                cutoff = train_positive.quantile(0.9)
                y = (frame.loc[train, "pnl"] >= cutoff)
                if y.nunique() < 2:
                    continue
                estimator = lgb.LGBMClassifier(scale_pos_weight=float((~y).sum() / max(1, y.sum())), **FIT)
                estimator.fit(X, y)
            elif kind == "huber":
                estimator = lgb.LGBMRegressor(objective="huber", **FIT)
                estimator.fit(X, frame.loc[train, "pnl"], sample_weight=sample_weight)
            elif kind == "tweedie":
                estimator = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, **FIT)
                estimator.fit(X, frame.loc[train, "win_size"])
            fitted = True
        if not fitted or estimator is None:
            continue
        X_test = frame.loc[test_mask, columns]
        if kind in {"clf", "bigwin"}:
            score.loc[test_mask] = estimator.predict_proba(X_test)[:, 1]
        else:
            score.loc[test_mask] = estimator.predict(X_test)
    return score, time.time() - t0


def report(score, label, elapsed):
    valid = score.notna()
    data = frame.loc[valid].copy()
    data["score"] = score[valid]
    baseline = -data["pnl"].sum()
    winner_dollars = data.loc[data["label_wins"], "pnl"].sum()

    auc = _rank_discrimination(data["score"].to_numpy(), data["label_wins"].to_numpy())["roc_auc"]
    # Out-of-sample AUC this high is not a good model, it is a leak. The honest
    # ceiling on this problem has been ~0.76 across every prior run.
    if auc > 0.95:
        raise RuntimeError(
            f"{label}: out-of-sample AUC {auc:.4f} indicates label leakage, not skill")
    # Dollar-weighted AUC: the ranking metric that matches the decision. Row AUC
    # rewards being right about trivial rows; this rewards being right where the
    # money is.
    order = data["score"].rank(pct=True)
    dollar_auc = float(
        (order * data["abs_pnl"]).loc[data["label_wins"]].sum() / max(1e-9, data.loc[data["label_wins"], "abs_pnl"].sum())
    )
    lines = []
    for pct in (0.05, 0.10, 0.20):
        hedged = data["score"] >= data["score"].quantile(1 - pct)
        firm = float(-data.loc[~hedged, "pnl"].sum())
        captured = data.loc[hedged & data["label_wins"], "pnl"].sum() / max(1e-9, winner_dollars)
        lines.append(f"      hedge {int(pct*100):>2}%: firm ${firm:>13,.0f}  "
                     f"vs flat {firm - baseline:>+13,.0f}  vs random {firm - baseline*(1-pct):>+13,.0f}  "
                     f"winner-$ captured {captured:>5.1%}")
    print(f"  {label:<34} AUC {auc:.4f} | dollar-AUC {dollar_auc:.4f} [{elapsed:.0f}s]")
    print("\n".join(lines), flush=True)
    return baseline


print("=== six objectives, judged on FIRM DOLLARS ===")
print("(vs random > 0 is the only bar that matters: it means the RANKING beat "
      "hedging an arbitrary same-sized subset)\n")
baseline = None
for kind, weight, label in (
    ("clf", None, "A unweighted classifier"),
    ("clf", "abs", "B classifier x |P&L|"),
    ("clf", "log", "C classifier x log1p|P&L|"),
    ("bigwin", None, "D BIG-WIN classifier (top decile)"),
    ("huber", None, "E Huber regression on P&L"),
    ("tweedie", None, "F Tweedie regression on win size"),
):
    score, elapsed = walk_forward(kind, weight, label)
    if score.notna().any():
        baseline = report(score, label, elapsed)
    else:
        print(f"  {label:<34} produced no scores", flush=True)
print(f"\nflat B-book across all servers: ${baseline:,.0f}")

