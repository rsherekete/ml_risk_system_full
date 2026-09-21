"""What actually causes the firm's drawdown? Everything follows from the answer.

The equity curve is not smooth: it carries a $31.3M drawdown against $1.096bn of
profit. Smoothing that is worth far more than raising the average, because the
average is already excellent -- clients lose ~$250-300 per account-day and no
routing policy improves that by more than about 1%.

So the real objective is not "predict which client wins". It is "stop the days
that hurt". And those are structurally different problems:

  * If the drawdown is ONE WHALE winning, the answer is account-level: find that
    account and hedge it. That is what the current model tries to do.
  * If the drawdown is MANY CLIENTS CORRELATED -- everyone long gold when gold
    rips -- then no per-account model can see it, because each account looks
    individually unremarkable. The risk lives in the NET BOOK, and the answer is
    a portfolio layer that hedges concentrated exposure.

A per-account model cannot fix a portfolio problem, and vice versa. This
measures which one the firm actually has.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

frame = ms.load_scores(ms.VIEW_TRADING)
frame["day"] = pd.to_datetime(frame["day"])

daily = frame.groupby("day")["pnl"].sum()
firm = -daily                      # firm earns the negative of client P&L
curve = firm.cumsum()
drawdown = curve - curve.cummax()

trough = drawdown.idxmin()
peak = curve.loc[:trough].idxmax()
recovery = curve.loc[trough:]
recovered = recovery[recovery >= curve.loc[peak]]
recovered_on = recovered.index[0] if len(recovered) else None

print("=== THE DRAWDOWN ===")
print(f"  worst drawdown : ${drawdown.min():,.0f}")
print(f"  peak           : {peak.date()}  (cumulative ${curve.loc[peak]:,.0f})")
print(f"  trough         : {trough.date()}  (cumulative ${curve.loc[trough]:,.0f})")
print(f"  duration       : {(trough - peak).days} days")
print(f"  recovered on   : {recovered_on.date() if recovered_on is not None else 'not yet'}"
      f"{f'  ({(recovered_on - trough).days} days later)' if recovered_on is not None else ''}")

window = firm.loc[peak:trough]
print(f"\n  days in the fall: {len(window)} | losing days: {(window < 0).sum()}")
print(f"  worst 5 days:")
for day, value in window.nsmallest(5).items():
    print(f"    {day.date()}  ${value:>14,.0f}")
print(f"  those 5 days are {window.nsmallest(5).sum() / window[window < 0].sum():.1%} "
      f"of all losses in the fall")

# ---- concentration: whale or crowd? ---------------------------------------
fall = frame.loc[(frame["day"] >= peak) & (frame["day"] <= trough)].copy()
by_account = fall.groupby("account_key")["pnl"].sum().sort_values(ascending=False)
winners = by_account[by_account > 0]

print("\n=== WHALE OR CROWD? ===")
print(f"  accounts active in the fall : {len(by_account):,}")
print(f"  accounts that WON           : {len(winners):,}")
print(f"  total client winnings       : ${winners.sum():,.0f}")
for n in (1, 5, 10, 50, 100, 500):
    if n <= len(winners):
        share = winners.nlargest(n).sum() / winners.sum()
        print(f"    top {n:>4} winners = {share:6.1%} of all client winnings")

print("\n  interpretation:")
top10 = winners.nlargest(10).sum() / winners.sum() if len(winners) else 0
if top10 > 0.5:
    print("    CONCENTRATED -- a handful of accounts. An account-level model can")
    print("    address this directly, and should be able to.")
else:
    print("    DISPERSED -- thousands of ordinary accounts winning together, which")
    print("    is the signature of CORRELATED market exposure, not of skill. No")
    print("    per-account model can see this: each account looks unremarkable on")
    print("    its own. It needs a portfolio layer on net book exposure.")

# ---- did the model see it coming? -----------------------------------------
print("\n=== DID THE MODEL SEE IT? ===")
scored = fall.loc[np.isfinite(fall["score"].to_numpy(dtype="float64"))]
top_winners = set(winners.nlargest(max(1, len(winners) // 100)).index)
flag = scored["account_key"].isin(top_winners)
if flag.any() and (~flag).any():
    print(f"  mean score, top-1% winners : {scored.loc[flag, 'score'].mean():.4f}")
    print(f"  mean score, everyone else  : {scored.loc[~flag, 'score'].mean():.4f}")
    caught = scored.loc[flag, "score"] >= 0.80
    print(f"  top-1% winner rows scored >= 0.80: {caught.mean():.1%} "
          f"(the live hedging rule)")

# ---- how much of total profit comes from how few days? --------------------
print("\n=== SHAPE OF THE PROFIT ===")
print(f"  total firm profit: ${firm.sum():,.0f} over {len(firm)} days")
print(f"  losing days: {(firm < 0).sum()} ({(firm < 0).mean():.1%})")
print(f"  sum of losing days:  ${firm[firm < 0].sum():,.0f}")
print(f"  sum of winning days: ${firm[firm > 0].sum():,.0f}")
print(f"  removing ALL losing days would lift profit by "
      f"{-firm[firm < 0].sum() / firm.sum():.1%}")
worst20 = firm.nsmallest(20).sum()
print(f"  the worst 20 days alone cost ${worst20:,.0f} "
      f"({-worst20 / firm.sum():.1%} of total profit)")
