"""Does ROC AUC 0.76 translate into profitable hedging? Measure, don't assume.

AUC measures ORDERING only: given a random winner and a random loser, how
often is the winner ranked higher. It says nothing about whether a predicted
0.70 actually wins 70% of the time -- and hedging P&L depends entirely on that,
because the value of hedging one account-day is:

    (2 * P(win) - 1) * |pnl|      positive only where P(win) > 0.5 genuinely

So this reports, per predicted-probability decile:
  * actual win rate        -- is the model calibrated, or just well-ordered?
  * mean client P&L        -- do the predicted winners carry money?
  * hedging P&L            -- what hedging that decile would actually have paid
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
MIN_TRAIN, CADENCE = 20, 5

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
frame["label_wins"] = frame["target_client_wins"].astype(bool)
frame["label_pnl"] = frame["target_profit"]
days = sorted(frame["decision_day"].unique())

p = pd.Series(np.nan, index=frame.index, dtype="float64")
clf = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)
ready = False
t0 = time.time()
for offset in range(MIN_TRAIN, len(days)):
    test_mask = frame["decision_day"] == days[offset]
    if not test_mask.any():
        continue
    if (offset - MIN_TRAIN) % CADENCE == 0 or not ready:
        train = frame["decision_day"].isin(days[:offset])
        y = frame.loc[train, "label_wins"]
        if y.notna().sum() > 100 and y.nunique() >= 2:
            clf.fit(frame.loc[train, columns], y); ready = True
    if ready:
        p.loc[test_mask] = clf.predict_proba(frame.loc[test_mask, columns])[:, 1]

valid = p.notna()
scored = frame.loc[valid, ["account_key", "label_wins", "label_pnl"]].copy()
scored["p_win"] = p[valid]
auc = _rank_discrimination(scored["p_win"].to_numpy(), scored["label_wins"].to_numpy())["roc_auc"]
print(f"scored {len(scored):,} rows, ROC AUC {auc:.4f}, base win rate {scored['label_wins'].mean():.3f} [{time.time()-t0:.0f}s]\n")

print(f"predicted p_win range: {scored['p_win'].min():.3f} to {scored['p_win'].max():.3f}, "
      f"mean {scored['p_win'].mean():.3f}, std {scored['p_win'].std():.3f}\n")

scored["decile"] = pd.qcut(scored["p_win"], 10, labels=False, duplicates="drop")
report = scored.groupby("decile").agg(
    n=("label_wins", "size"),
    mean_predicted=("p_win", "mean"),
    actual_win_rate=("label_wins", "mean"),
    mean_client_pnl=("label_pnl", "mean"),
    total_client_pnl=("label_pnl", "sum"),
).reset_index()
report["calibration_gap"] = report["mean_predicted"] - report["actual_win_rate"]
# Hedging a decile pays exactly the clients' P&L in it: + when they won.
report["hedging_pnl"] = report["total_client_pnl"]
print("per predicted-probability decile (9 = most confident the client WINS):")
print(report.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

print("\nKEY QUESTION: in the top decile, do clients actually win more than they lose in DOLLARS?")
top = scored.loc[scored["decile"] == report["decile"].max()]
print(f"  top decile: {top['label_wins'].mean():.1%} win rate, "
      f"mean P&L ${top['label_pnl'].mean():,.0f}, total ${top['label_pnl'].sum():,.0f}")
print(f"  -> hedging the top decile would have {'SAVED' if top['label_pnl'].sum() > 0 else 'COST'} "
      f"${abs(top['label_pnl'].sum()):,.0f}")

print("\nwin rate vs dollar outcome, by decile (the gap between them is the whole story):")
for _, row in report.iterrows():
    direction = "clients WON $" if row["total_client_pnl"] > 0 else "clients LOST $"
    print(f"  decile {int(row['decile'])}: predicted {row['mean_predicted']:.1%}, actual {row['actual_win_rate']:.1%}, "
          f"{direction}{abs(row['total_client_pnl']):>14,.0f}")
