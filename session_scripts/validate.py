"""Two questions the app's headline numbers depend on, neither yet answered.

1. RECONCILIATION. The artefact says flat B-book earned $1.096bn over two years.
   That is the single largest number on the site and everything else is measured
   against it. If the exposure-day expansion double-counted P&L it would be
   inflated and nobody would notice, so it is checked against the warehouse's
   own sum of net_profit -- an independent path to the same quantity.

2. STABILITY OUT OF SAMPLE. The walk-forward AUC of 0.5812 is an average over
   730 days. An average can hide a model that worked in 2024 and stopped
   working, which is the failure mode that matters: the desk would be running
   it today. So the same curve is recomputed on the most recent slices alone.
   Nothing here retrains -- these are the stored walk-forward scores, each
   produced by a model fitted only on days before it.
"""
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import data_store, model_service as ms

frame = ms.load_scores(ms.VIEW_TRADING)
frame["day"] = pd.to_datetime(frame["day"])
print(f"artefact: {len(frame):,} rows | {frame['day'].min().date()} -> "
      f"{frame['day'].max().date()} | {frame['day'].nunique()} days")

# ---------------------------------------------------------------- 1. reconcile
artefact_pnl = float(frame["pnl"].sum())
end = frame["day"].max().to_pydatetime().replace(tzinfo=timezone.utc)
start = end - timedelta(days=730)

warehouse_pnl, warehouse_rows = 0.0, 0
for server in sorted(p.name for p in data_store.WAREHOUSE.iterdir() if p.is_dir()):
    chunk = data_store.read_history(databases=(server,), start=start, end=end,
                                    columns=["database", "net_profit", "close_time"])
    if chunk.empty:
        continue
    # The artefact books P&L on the CLOSING day, so only closed trades count.
    closed = chunk.loc[chunk["close_time"].notna() & chunk["net_profit"].notna()]
    warehouse_pnl += float(closed["net_profit"].sum())
    warehouse_rows += len(closed)
    del chunk, closed

gap = artefact_pnl - warehouse_pnl
rel = abs(gap) / max(abs(warehouse_pnl), 1.0)
print(f"\n--- reconciliation (client P&L; broker earns the negative) ---")
print(f"  artefact  sum(pnl)        {artefact_pnl:>20,.0f}")
print(f"  warehouse sum(net_profit) {warehouse_pnl:>20,.0f}  ({warehouse_rows:,} closed trades)")
print(f"  difference                {gap:>20,.0f}   ({rel:.2%})")
print(f"  VERDICT: {'MATCH' if rel < 0.01 else 'MISMATCH -- headline figure is not trustworthy'}")

# --------------------------------------------------------------- 2. stability
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

cfg = ms.load_config(ms.VIEW_TRADING)
max_day = frame["day"].max()

print(f"\n--- out-of-sample stability by recency (hedge {cfg.hedge_fraction:.0%}) ---")
print(f"{'window':<14}{'days':>6}{'rows':>12}{'AUC':>8}"
      f"{'flat $':>16}{'model $':>16}{'flat DD':>15}{'model DD':>15}")

for label, days in (("last 90d", 90), ("last 180d", 180), ("last 365d", 365),
                    ("full 730d", 730)):
    window = frame.loc[frame["day"] > max_day - pd.Timedelta(days=days)]
    if window.empty:
        continue
    curve = ms.equity_curves(window, cfg.hedge_fraction, cfg.probability_threshold)

    auc = np.nan
    if roc_auc_score is not None:
        # Reconstruct the label the model was scored against: did the account
        # lose more than half its own sigma over the horizon? Sign convention
        # matches training -- a positive label is a client loss (firm gain).
        label_col = (window["pnl"] < 0).to_numpy()
        scores = window["score"].to_numpy(dtype="float64")
        keep = np.isfinite(scores) & (window["sigma"].to_numpy() > 0)
        if keep.sum() > 1000 and len(np.unique(label_col[keep])) == 2:
            auc = roc_auc_score(label_col[keep], scores[keep])

    print(f"{label:<14}{len(curve):>6}{len(window):>12,}{auc:>8.4f}"
          f"{curve['flat'].sum():>16,.0f}{curve['model'].sum():>16,.0f}"
          f"{curve['flat_dd'].min():>15,.0f}{curve['model_dd'].min():>15,.0f}")

# Did the trade-off hold recently, or only early?
recent = frame.loc[frame["day"] > max_day - pd.Timedelta(days=180)]
print(f"\n--- does the drawdown benefit survive in the last 180 days? ---")
print(f"{'hedge':<10}{'profit':>18}{'vs flat':>16}{'maxDD':>16}{'vs flat':>15}{'Sharpe':>8}")
base = ms.equity_curves(recent, 1e-6, 0.0)
flat_p, flat_d = base["flat"].sum(), base["flat_dd"].min()
print(f"{'FLAT':<10}{flat_p:>18,.0f}{'':>16}{flat_d:>16,.0f}{'':>15}"
      f"{base['flat'].mean() / max(base['flat'].std(), 1e-9) * np.sqrt(252):>8.2f}")
for fraction in (0.02, 0.05, 0.10, 0.20):
    curve = ms.equity_curves(recent, fraction, 0.0)
    profit, dd = curve["model"].sum(), curve["model_dd"].min()
    sharpe = curve["model"].mean() / max(curve["model"].std(), 1e-9) * np.sqrt(252)
    flag = "  <-- BEATS BOTH" if (profit > flat_p and dd > flat_d) else ""
    print(f"{fraction:<10.0%}{profit:>18,.0f}{profit - flat_p:>+16,.0f}"
          f"{dd:>16,.0f}{dd - flat_d:>+15,.0f}{sharpe:>8.2f}{flag}")
