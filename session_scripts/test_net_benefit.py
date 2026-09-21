"""NET benefit of hedging, not one-sided cost capture.

The capture metric used earlier only counted money lost to winning clients. It
gave a free pass to hedging an account that then LOSES -- which costs the firm
the B-book profit it would otherwise have earned. Ranking purely by predicted
size scores well on that metric precisely because it does not pay for its
mistakes.

Correct accounting, per account-day:
    B-book P&L = -client_pnl        (firm wins when the client loses)
    A-book P&L =  0                 (hedged, ignoring markup)
    benefit of hedging = 0 - (-client_pnl) = client_pnl

So hedging pays exactly when the client wins, and costs exactly when they
lose. Summing `client_pnl` over the hedged set gives the true P&L impact --
and a ranking that hedges big losers is now penalised for it.
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from trading_data.behaviour_features import build_active_day_frame, feature_columns
from trading_data.bigquery_data_client import compact_memory

RECORDS = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"
MIN_TRAIN, CADENCE = 20, 5

parts = []
for database in sorted(pd.read_parquet(RECORDS, columns=["database"])["database"].unique()):
    part = compact_memory(pd.read_parquet(RECORDS, filters=[("database", "==", database)]))
    parts.append(build_active_day_frame(part, max_gap_days=None))
    del part; gc.collect()
frame = pd.concat(parts, ignore_index=True)
del parts; gc.collect()

columns = feature_columns(frame)
for column in columns:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
frame["decision_day"] = pd.to_datetime(frame["decision_day"]).dt.normalize()
frame["label_wins"] = frame["target_client_wins"].astype(bool)
frame["label_client_pnl"] = frame["target_profit"]          # + = client won = hedging pays
frame["label_firm_cost"] = frame["target_profit"].clip(lower=0)
frame["label_log_win_size"] = np.log1p(frame["label_firm_cost"])
frame["label_abs_pnl"] = frame["target_profit"].abs()
days = sorted(frame["decision_day"].unique())

p_win = pd.Series(np.nan, index=frame.index, dtype="float64")
size = pd.Series(np.nan, index=frame.index, dtype="float64")
abs_size = pd.Series(np.nan, index=frame.index, dtype="float64")
clf = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)
reg = lgb.LGBMRegressor(n_estimators=200, verbose=-1, random_state=0)
reg_abs = lgb.LGBMRegressor(n_estimators=200, verbose=-1, random_state=0)
ok = [False, False, False]

t0 = time.time()
for offset in range(MIN_TRAIN, len(days)):
    test_mask = frame["decision_day"] == days[offset]
    if not test_mask.any():
        continue
    if (offset - MIN_TRAIN) % CADENCE == 0 or not all(ok):
        train_mask = frame["decision_day"].isin(days[:offset])
        y = frame.loc[train_mask, "label_wins"]
        if y.notna().sum() > 100 and y.nunique() >= 2:
            clf.fit(frame.loc[train_mask, columns], y); ok[0] = True
        winners = train_mask & frame["label_wins"]
        if winners.sum() > 100:
            reg.fit(frame.loc[winners, columns], frame.loc[winners, "label_log_win_size"]); ok[1] = True
        if train_mask.sum() > 100:
            reg_abs.fit(frame.loc[train_mask, columns], np.log1p(frame.loc[train_mask, "label_abs_pnl"])); ok[2] = True
    if ok[0]:
        p_win.loc[test_mask] = clf.predict_proba(frame.loc[test_mask, columns])[:, 1]
    if ok[1]:
        size.loc[test_mask] = reg.predict(frame.loc[test_mask, columns])
    if ok[2]:
        abs_size.loc[test_mask] = reg_abs.predict(frame.loc[test_mask, columns])

valid = p_win.notna() & size.notna() & abs_size.notna()
scored = frame.loc[valid].copy()
scored["p_win"] = p_win[valid]
scored["expected_size"] = np.expm1(size[valid]).clip(lower=0)
scored["expected_abs"] = np.expm1(abs_size[valid]).clip(lower=0)
# Expected NET benefit of hedging: E[client_pnl] = P(win)*E[win] - P(lose)*E[loss].
# Using the magnitude model for both sides via the |pnl| model.
scored["expected_net_benefit"] = (2 * scored["p_win"] - 1) * scored["expected_abs"]
scored["priority_pw_x_size"] = scored["p_win"] * scored["expected_size"]
print(f"scored {len(scored):,} rows [{time.time()-t0:.0f}s]", flush=True)

total_pnl = scored["label_client_pnl"].sum()
print(f"\nIf the firm B-booked EVERYTHING in this window it would net "
      f"${-total_pnl:,.0f} from these accounts.")
print("(so hedging the whole book would forgo that -- the question is whether a SUBSET beats it)\n")

print(f"{'ranking':<26}" + "".join(f"{f'top {int(p*100)}%':>14}" for p in (0.05, 0.10, 0.20, 0.30)))
print(f"{'':<26}" + "".join(f"{'net $ hedged':>14}" for _ in range(4)))
rankings = {
    "expected net benefit": scored["expected_net_benefit"],
    "P(win) x E[size|win]": scored["priority_pw_x_size"],
    "size alone": scored["expected_size"],
    "P(win) alone": scored["p_win"],
    "random": pd.Series(np.random.default_rng(0).random(len(scored)), index=scored.index),
}
for label, series in rankings.items():
    cells = []
    for pct in (0.05, 0.10, 0.20, 0.30):
        cutoff = series.quantile(1 - pct)
        hedged = scored.loc[series >= cutoff]
        cells.append(f"{hedged['label_client_pnl'].sum():>13,.0f}")
    print(f"  {label:<24}" + "".join(cells), flush=True)

print("\nPositive = hedging that subset SAVED the firm money (clients won).")
print("Negative = hedging it COST the firm money (those clients would have lost).")
