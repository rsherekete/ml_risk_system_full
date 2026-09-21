"""Where does the trading model actually fail? Four measurements, no tuning.

The 90-day model returned +5.6% profit and -21.5% drawdown against a flat
B-book. The two-year model returns +1.0% and -2.2% on the same algorithm, so
something structural changed rather than something numeric. Three things did:

  1. an active day became an EXPOSURE day (opened, closed, OR carried), which
     added a large population of rows that realise no P&L at all;
  2. the training window became a rolling 365 days;
  3. the horizon counts exposure days rather than trading days.

And one earlier experiment is unexplained: ranking by expected dollars lost
$411M, which can only happen if the model fails precisely where the money is.

Q1  How much of the decision population can produce no P&L at all?
Q2  Is the model predictive on LARGE accounts, or only on small ones?
Q3  Where is the hedge quota actually being spent?
Q4  Does every account size band contribute firm profit, or only some?
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

frame = ms.load_scores(ms.VIEW_TRADING)
frame["day"] = pd.to_datetime(frame["day"])
config = ms.load_config(ms.VIEW_TRADING)
threshold = config.sigma_threshold

sigma = frame["sigma"].to_numpy(dtype="float64")
score = frame["score"].to_numpy(dtype="float64")
pnl = frame["pnl"].to_numpy(dtype="float64")
sane = np.isfinite(sigma) & np.isfinite(score) & (np.abs(sigma) <= 100)
label = sigma > threshold

print(f"rows {len(frame):,} | days {frame['day'].nunique()}")

# ---- Q1: dead rows ---------------------------------------------------------
zero = pnl == 0
print(f"\nQ1  rows that realise NO P&L: {zero.sum():,} ({zero.mean():.1%})")
print(f"    every hedge spent on one of these costs nothing and buys nothing,")
print(f"    but it still consumes the quota.")
if "carried" in frame.columns:
    carried = pd.to_numeric(frame["carried"], errors="coerce").fillna(0) > 0
    print(f"    of which flagged as carrying days: {(zero & carried).sum():,}")

# ---- Q2: is the model predictive where the money is? -----------------------
# Size = the account's own average absolute daily P&L over the sample. Used only
# to BUCKET rows for diagnosis, never as a feature or a decision input.
size = frame.groupby("account_key")["pnl"].transform(lambda s: s.abs().mean())
size = pd.to_numeric(size, errors="coerce").to_numpy()
ok = sane & np.isfinite(size) & (size > 0)

deciles = pd.qcut(pd.Series(size[ok]), 10, labels=False, duplicates="drop")
print(f"\nQ2/Q4  by account size decile (10 = largest)")
print(f"{'decile':<8}{'rows':>12}{'avg size $':>12}{'AUC':>8}"
      f"{'firm P&L':>16}{'share':>8}{'base rate':>11}")
sub = pd.DataFrame({
    "decile": deciles.to_numpy(), "score": score[ok],
    "label": label[ok], "pnl": pnl[ok], "size": size[ok]})
total_firm = -sub["pnl"].sum()
for decile, group in sub.groupby("decile"):
    auc = np.nan
    if roc_auc_score is not None and group["label"].nunique() == 2 and len(group) > 1000:
        auc = roc_auc_score(group["label"], group["score"])
    firm = -group["pnl"].sum()
    print(f"{int(decile) + 1:<8}{len(group):>12,}{group['size'].mean():>12,.0f}"
          f"{auc:>8.4f}{firm:>16,.0f}{firm / total_firm:>8.1%}"
          f"{group['label'].mean():>11.3f}")

# ---- Q3: where does the quota go? -----------------------------------------
# Reproduce the live selection and ask what it actually bought.
print(f"\nQ3  where the hedge quota is spent (threshold {config.probability_threshold})")
for name, rule in (("prob >= 0.80", 0.80), ("prob >= 0.90", 0.90)):
    chosen = sane & (score >= rule)
    if not chosen.any():
        continue
    chosen_zero = (pnl == 0) & chosen
    print(f"  {name}: {chosen.sum():,} hedged rows, "
          f"{chosen_zero.sum() / max(1, chosen.sum()):.1%} of them realise no P&L")
    print(f"      firm P&L forgone: {-pnl[chosen].sum():,.0f} "
          f"(negative = hedging AVOIDED a loss)")

# ---- what a perfect model would earn: the ceiling --------------------------
# Hedge exactly the account-days where the client actually won. Nothing can beat
# this, and it says whether the remaining upside is worth chasing at all.
winners = pnl > 0
flat = -pnl.sum()
oracle = -pnl[~winners].sum()
print(f"\nCEILING  flat B-book: {flat:,.0f}")
print(f"         perfect hindsight (hedge every client win): {oracle:,.0f}")
print(f"         maximum possible uplift: {oracle - flat:,.0f} "
      f"({(oracle - flat) / abs(flat):.1%})")
print(f"         current best policy captures ~11.2M of that "
      f"({11_170_907 / max(1, oracle - flat):.2%} of the ceiling)")
