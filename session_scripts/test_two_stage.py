"""Two-stage A-book priority: P(win) x E[|pnl| | win].

The single-stage rank target failed (Spearman -0.033) because it ranked ALL
accounts by |P&L|, mixing winners and losers into one ordering. The decision
actually being made is narrower: among accounts expected to WIN, which would
cost the firm most if left on the B-book? That is a conditional magnitude
question, and conditioning on winning is a far easier target than
unconditional P&L.

  priority = P(client wins) x E[|pnl| | client wins]

Stage 1 is the classifier already measured at ROC AUC ~0.76.
Stage 2 is fitted ONLY on rows where the client actually won, so it never has
to explain the sign -- only the size, given the sign.

Scored two ways, because AUC is the wrong metric for a priority ordering:
  * Spearman of predicted priority vs realised firm cost (-pnl where client won)
  * Top-decile capture: of the total dollars the firm would lose to winning
    clients, what share sits in the accounts the model ranked highest? This is
    the number that matters operationally -- hedging capacity is limited, so
    what counts is whether the top of the list is where the money is.
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

PATH = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"
MIN_TRAIN, CADENCE = 20, 5

parts = []
for database in sorted(pd.read_parquet(PATH, columns=["database"])["database"].unique()):
    part = compact_memory(pd.read_parquet(PATH, filters=[("database", "==", database)]))
    parts.append(build_active_day_frame(part, max_gap_days=None))
    del part; gc.collect()
frame = pd.concat(parts, ignore_index=True)
del parts; gc.collect()

columns = feature_columns(frame)
for column in columns:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
frame["decision_day"] = pd.to_datetime(frame["decision_day"]).dt.normalize()
frame["label_wins"] = frame["target_client_wins"].astype(bool)
# What the firm loses by B-booking a winner: the client's win IS the firm's loss.
frame["label_firm_cost"] = frame["target_profit"].clip(lower=0)
# Magnitude target, conditional on winning. Log1p because win sizes are heavy
# tailed and the ordering matters far more than the raw scale.
frame["label_log_win_size"] = np.log1p(frame["label_firm_cost"])
days = sorted(frame["decision_day"].unique())
print(f"{len(frame):,} rows, {len(columns)} features, {len(days)} days, win rate {frame['label_wins'].mean():.3f}", flush=True)

p_win = pd.Series(np.nan, index=frame.index, dtype="float64")
magnitude = pd.Series(np.nan, index=frame.index, dtype="float64")
clf = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)
reg = lgb.LGBMRegressor(n_estimators=200, verbose=-1, random_state=0)
clf_ready = reg_ready = False

t0 = time.time()
for offset in range(MIN_TRAIN, len(days)):
    test_mask = frame["decision_day"] == days[offset]
    if not test_mask.any():
        continue
    if (offset - MIN_TRAIN) % CADENCE == 0 or not (clf_ready and reg_ready):
        train_mask = frame["decision_day"].isin(days[:offset])
        y = frame.loc[train_mask, "label_wins"]
        if y.notna().sum() > 100 and y.nunique() >= 2:
            clf.fit(frame.loc[train_mask, columns], y)
            clf_ready = True
        # Stage 2 sees ONLY winning rows -- it never learns the sign.
        winner_mask = train_mask & frame["label_wins"]
        if winner_mask.sum() > 100:
            reg.fit(frame.loc[winner_mask, columns], frame.loc[winner_mask, "label_log_win_size"])
            reg_ready = True
    if clf_ready:
        p_win.loc[test_mask] = clf.predict_proba(frame.loc[test_mask, columns])[:, 1]
    if reg_ready:
        magnitude.loc[test_mask] = reg.predict(frame.loc[test_mask, columns])

valid = p_win.notna() & magnitude.notna()
scored = frame.loc[valid].copy()
scored["p_win"] = p_win[valid]
scored["expected_size"] = np.expm1(magnitude[valid]).clip(lower=0)
scored["priority"] = scored["p_win"] * scored["expected_size"]
print(f"scored {len(scored):,} rows [{time.time()-t0:.0f}s]\n", flush=True)

disc = _rank_discrimination(scored["p_win"].to_numpy(), scored["label_wins"].to_numpy())
print(f"stage 1  P(win)                ROC AUC {disc['roc_auc']:.4f}")

winners = scored.loc[scored["label_wins"]]
sp_mag = float(pd.Series(winners["expected_size"].to_numpy()).rank().corr(winners["label_firm_cost"].rank()))
print(f"stage 2  E[size | win]         Spearman {sp_mag:+.4f}  (on {len(winners):,} actual winners)")

sp_pri = float(pd.Series(scored["priority"].to_numpy()).rank().corr(scored["label_firm_cost"].rank()))
print(f"combined priority             Spearman {sp_pri:+.4f}  (vs firm cost, all rows)")

print("\ntop-decile capture -- share of total firm cost sitting in the top N% by each ranking:")
total_cost = scored["label_firm_cost"].sum()
for label, series in (("priority P(win)xsize", scored["priority"]),
                      ("P(win) alone", scored["p_win"]),
                      ("size alone", scored["expected_size"]),
                      ("random", pd.Series(np.random.default_rng(0).random(len(scored)), index=scored.index))):
    row = []
    for pct in (0.05, 0.10, 0.20):
        cutoff = series.quantile(1 - pct)
        captured = scored.loc[series >= cutoff, "label_firm_cost"].sum() / total_cost
        row.append(f"top {int(pct*100):>2}%: {captured:6.1%}")
    print(f"  {label:<22} " + "   ".join(row))
print("\n(random is the baseline -- top 10% of accounts holds 10% of cost if ranking is useless)")
