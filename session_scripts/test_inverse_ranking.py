"""Use the ranking INVERSELY: which accounts to fade hardest, not hedge.

The calibration table showed something counterintuitive: the highest-win-rate
deciles are the MOST profitable to B-book. Decile 8 clients win 81% of the
time and still yield the firm $554 per account-day -- more than any other
decile -- while decile 0 (92% losers) yields only $62.

So the model's ranking is informative but inverted for this purpose: it
identifies who to take MORE of the other side of, not who to hedge away. This
tests whether amplifying B-book exposure to the top of that inverted ranking
beats flat B-booking everything, and critically what it does to drawdown --
amplifying exposure amplifies both tails.
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from trading_data.behaviour_features import build_active_day_frame, feature_columns
from trading_data.bigquery_data_client import compact_memory

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

# Predict the firm's P&L directly (-client_pnl): positive = profitable to fade.
frame["label_firm_pnl"] = -frame["target_profit"]
frame["label_signed_log"] = np.sign(frame["label_firm_pnl"]) * np.log1p(frame["label_firm_pnl"].abs())

pred_firm = pd.Series(np.nan, index=frame.index, dtype="float64")
reg = lgb.LGBMRegressor(n_estimators=200, verbose=-1, random_state=0)
ready = False
t0 = time.time()
for offset in range(MIN_TRAIN, len(days)):
    test_mask = frame["decision_day"] == days[offset]
    if not test_mask.any():
        continue
    if (offset - MIN_TRAIN) % CADENCE == 0 or not ready:
        train = frame["decision_day"].isin(days[:offset])
        if train.sum() > 100:
            reg.fit(frame.loc[train, columns], frame.loc[train, "label_signed_log"]); ready = True
    if ready:
        pred_firm.loc[test_mask] = reg.predict(frame.loc[test_mask, columns])

valid = pred_firm.notna()
scored = frame.loc[valid, ["account_key", "decision_day", "label_pnl", "label_firm_pnl", "label_wins"]].copy()
scored["fade_score"] = pred_firm[valid]
print(f"scored {len(scored):,} rows [{time.time()-t0:.0f}s]")
spearman = float(pd.Series(scored["fade_score"].to_numpy()).rank().corr(scored["label_firm_pnl"].rank()))
print(f"Spearman(fade_score, realised firm P&L) = {spearman:+.4f}\n")

baseline = scored["label_firm_pnl"].sum()
print(f"flat B-book everything: ${baseline:,.0f}\n")

print("AMPLIFIED B-BOOK: take extra exposure to the top-ranked fade candidates")
print(f"{'strategy':<44}{'firm P&L':>16}{'vs flat':>15}{'max DD':>15}")
for pct in (0.05, 0.10, 0.20, 0.30):
    for multiplier in (1.5, 2.0):
        cutoff = scored["fade_score"].quantile(1 - pct)
        boosted = scored["fade_score"] >= cutoff
        weight = np.where(boosted, multiplier, 1.0)
        pnl = (scored["label_firm_pnl"] * weight)
        daily = pnl.groupby(scored["decision_day"]).sum().sort_index()
        equity = daily.cumsum()
        drawdown = float((equity - equity.cummax()).min())
        total = float(pnl.sum())
        print(f"  top {int(pct*100):>2}% at {multiplier}x{'':<28}{total:>16,.0f}{total-baseline:>+15,.0f}{drawdown:>15,.0f}")

flat_daily = scored["label_firm_pnl"].groupby(scored["decision_day"]).sum().sort_index()
flat_equity = flat_daily.cumsum()
print(f"\n  {'flat B-book (reference)':<42}{baseline:>16,.0f}{'--':>15}{float((flat_equity-flat_equity.cummax()).min()):>15,.0f}")

print("\nDECILE CHECK: realised firm P&L per account-day, by fade_score decile")
scored["decile"] = pd.qcut(scored["fade_score"], 10, labels=False, duplicates="drop")
report = scored.groupby("decile").agg(
    n=("label_pnl", "size"), win_rate=("label_wins", "mean"),
    mean_firm_pnl=("label_firm_pnl", "mean"), total_firm_pnl=("label_firm_pnl", "sum"),
)
print(report.to_string(float_format=lambda v: f"{v:,.2f}"))
print("\n(if the model ranks fade candidates well, mean_firm_pnl should rise with decile)")
