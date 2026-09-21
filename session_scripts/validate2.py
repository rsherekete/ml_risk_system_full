"""Reconciliation and stability, with two flaws in the first pass corrected.

FLAW 1 -- the windows did not match. The artefact covers 2024-10-30 to
2026-08-30 (670 days); the warehouse sum was taken over 730. Sixty extra days
of client losses were charged against the artefact as if they were missing,
which is a difference in the question asked, not in the answer. The warehouse
is now summed over exactly the artefact's own date range.

FLAW 2 -- the label was wrong. AUC was computed against the sign of SAME-DAY
P&L, but the model predicts FORWARD self-relative P&L over the next five
exposure days. Scoring a forward model on a contemporaneous label measures
nothing; a value below 0.5 there is an artefact of the mistake, not evidence
the model is inverted. The real label is rebuilt here from the same definition
training uses.
"""
import sys
from datetime import timezone

import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import data_store, model_service as ms

frame = ms.load_scores(ms.VIEW_TRADING)
frame["day"] = pd.to_datetime(frame["day"])
lo, hi = frame["day"].min(), frame["day"].max()
print(f"artefact: {len(frame):,} rows | {lo.date()} -> {hi.date()} | {frame['day'].nunique()} days")

# ------------------------------------------------- 1. reconcile, matched window
start = lo.to_pydatetime().replace(tzinfo=timezone.utc)
end = (hi + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=timezone.utc)
artefact_pnl = float(frame["pnl"].sum())

warehouse_pnl, rows, dropped_open, dropped_pnl = 0.0, 0, 0, 0.0
for server in sorted(p.name for p in data_store.WAREHOUSE.iterdir() if p.is_dir()):
    chunk = data_store.read_history(databases=(server,), start=start, end=end,
                                    columns=["database", "net_profit", "open_time", "close_time"])
    if chunk.empty:
        continue
    closed = chunk.loc[chunk["close_time"].notna() & chunk["net_profit"].notna()]
    warehouse_pnl += float(closed["net_profit"].sum())
    rows += len(closed)
    # Trades the exposure builder discards for want of an open time -- a
    # candidate explanation for any residual gap, so it is measured not guessed.
    orphan = closed.loc[closed["open_time"].isna()]
    dropped_open += len(orphan)
    dropped_pnl += float(orphan["net_profit"].sum())
    del chunk, closed, orphan

gap = artefact_pnl - warehouse_pnl
rel = abs(gap) / max(abs(warehouse_pnl), 1.0)
print(f"\n--- reconciliation over the artefact's own window ---")
print(f"  artefact  sum(pnl)        {artefact_pnl:>20,.0f}")
print(f"  warehouse sum(net_profit) {warehouse_pnl:>20,.0f}  ({rows:,} closed trades)")
print(f"  difference                {gap:>20,.0f}   ({rel:.3%})")
print(f"  of which: trades with no open_time, dropped by the exposure builder")
print(f"            {dropped_open:>10,} trades{dropped_pnl:>26,.0f}")
residual = gap - (-dropped_pnl if dropped_pnl else 0.0)
print(f"  VERDICT: {'MATCH' if rel < 0.01 else 'gap exceeds 1% -- see breakdown above'}")

# ------------------------------------------------------- 2. the correct label
# Forward P&L over the next `horizon` exposure days for the account, relative to
# that account's own volatility. Shifted strictly into the future, per account.
cfg = ms.load_config(ms.VIEW_TRADING)
horizon = cfg.horizon_active_days
work = frame[["account_key", "day", "pnl", "score", "sigma"]].sort_values(
    ["account_key", "day"], kind="mergesort")
grouped = work.groupby("account_key", sort=False)["pnl"]
# Sum of the NEXT `horizon` rows: reverse-rolling, then shift so today is excluded.
forward = grouped.apply(
    lambda s: s.iloc[::-1].rolling(horizon, min_periods=1).sum().iloc[::-1].shift(-1)
).reset_index(level=0, drop=True)
work["forward"] = forward
sigma = work["sigma"].to_numpy(dtype="float64")
with np.errstate(divide="ignore", invalid="ignore"):
    relative = np.where(sigma > 0, work["forward"].to_numpy(dtype="float64") / sigma, np.nan)
# Positive class = the client is about to lose more than half a sigma (firm gain).
work["label"] = (relative < -cfg.sigma_threshold)
work["valid"] = np.isfinite(relative)

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

print(f"\n--- AUC on the ACTUAL target (forward {horizon} exposure days, "
      f"< -{cfg.sigma_threshold} sigma) ---")
print(f"{'window':<14}{'rows scored':>14}{'base rate':>11}{'AUC':>9}")
for label, days in (("last 90d", 90), ("last 180d", 180), ("last 365d", 365), ("full", 9999)):
    window = work.loc[work["day"] > hi - pd.Timedelta(days=days)]
    window = window.loc[window["valid"]]
    y = window["label"].to_numpy()
    s = window["score"].to_numpy(dtype="float64")
    keep = np.isfinite(s)
    if keep.sum() < 1000 or len(np.unique(y[keep])) < 2 or roc_auc_score is None:
        print(f"{label:<14}{keep.sum():>14,}{'--':>11}{'--':>9}")
        continue
    print(f"{label:<14}{keep.sum():>14,}{y[keep].mean():>11.3f}"
          f"{roc_auc_score(y[keep], s[keep]):>9.4f}")

# ------------------------------------------- 3. does the trade-off hold lately?
print(f"\n--- economics by window (does hedging still beat flat?) ---")
for label, days in (("last 90d", 90), ("last 180d", 180), ("last 365d", 365), ("full", 9999)):
    window = frame.loc[frame["day"] > hi - pd.Timedelta(days=days)]
    base = ms.equity_curves(window, 1e-6, 0.0)
    flat_p, flat_d = base["flat"].sum(), base["flat_dd"].min()
    winners = []
    for fraction in (0.02, 0.05, 0.10, 0.20):
        curve = ms.equity_curves(window, fraction, 0.0)
        if curve["model"].sum() > flat_p and curve["model_dd"].min() > flat_d:
            winners.append(f"{fraction:.0%}")
    print(f"  {label:<11} flat {flat_p:>15,.0f} / DD {flat_d:>13,.0f}   "
          f"dominating fractions: {', '.join(winners) if winners else 'NONE (trade-off only)'}")
