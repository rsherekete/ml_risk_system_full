"""TAF walk-forward abuse-cost analysis -- the real numbers for the proposal.

Framework:
  C (Toxic/Arbitrage)  = union of behavioural classes (rule_forecast masks)
  A (Abuse)            = C  intersect  {account made money over next n=5 days}
  Cost                 = realised USD extracted by C accounts on their winning
                         days (cent/contract-normalised; client gain = our loss)
  Recoverable          = the abuse-USD attributable to accounts a class model
                         would have flagged 5 days in advance (recall-weighted)
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from webapp import antifraud, rule_forecast

N = 5
frame = antifraud._frame().copy()
frame["decision_day"] = pd.to_datetime(frame["decision_day"])
frame = frame.sort_values(["account_key", "decision_day"])
pnl_col = next((c for c in ("realised_pnl", "day_pnl", "realized_pnl", "pnl") if c in frame.columns), None)
print("pnl col:", pnl_col, "| rows:", len(frame),
      "| days:", frame["decision_day"].min().date(), "->", frame["decision_day"].max().date(),
      "| accounts:", frame["account_key"].nunique())
frame["pnl"] = pd.to_numeric(frame[pnl_col], errors="coerce").fillna(0.0)

# forward n-day realised gain per account-day (next N rows) -- the ABUSE LABEL basis
g = frame.groupby("account_key")["pnl"]
frame["fwd_gain"] = g.transform(lambda s: s[::-1].rolling(N, min_periods=1).sum()
                                .shift(1)[::-1]).fillna(0.0)

masks = rule_forecast._masks(antifraud.load_rules())
n_days = frame["decision_day"].dt.normalize().nunique()
print(f"\ndistinct trading days: {n_days}\n")

# per-class C-membership, A-subset, realised cost
union = pd.Series(False, index=frame.index)
rows = []
for cls, (needed, build) in masks.items():
    if any(c not in frame.columns for c in needed):
        continue
    try:
        m = build(frame).fillna(False).to_numpy()
    except Exception:
        continue
    if m.mean() < 0.0005 or m.mean() > 0.6:
        continue
    union = union | pd.Series(m, index=frame.index)
    C = frame[m]
    # A = C accounts profitable over the forward window
    A = C[C["fwd_gain"] > 0]
    # realised cost = client winnings on C-days (non-overlapping, day-attributed)
    cost = float(C.loc[C["pnl"] > 0, "pnl"].sum())
    rows.append({"class": cls, "C_account_days": int(m.sum()),
                 "A_share_of_C": round(float((C["fwd_gain"] > 0).mean()), 3),
                 "A_accounts": int(A["account_key"].nunique()),
                 "cost_total_usd": cost, "cost_per_day_usd": cost / n_days})

byc = pd.DataFrame(rows).sort_values("cost_per_day_usd", ascending=False)
print("=== ABUSE COST BY CLASS (ordered by daily business impact) ===")
for r in byc.itertuples():
    print("  %-24s C-days=%7d  A/C=%4.0f%%  A-accts=%5d  cost=$%12s  /day=$%9s" % (
        r._1, r.C_account_days, r.A_share_of_C*100, r.A_accounts,
        f"{r.cost_total_usd:,.0f}", f"{r.cost_per_day_usd:,.0f}"))

# whole-universe totals
Cframe = frame[union.to_numpy()]
total_cost = float(Cframe.loc[Cframe["pnl"] > 0, "pnl"].sum())
total_client_win = float(frame.loc[frame["pnl"] > 0, "pnl"].sum())
print("\n=== UNIVERSE ===")
print("  total realised abuse cost (all C classes): $%,.0f  (=$%,.0f/day)" % (total_cost, total_cost/n_days))
print("  C accounts: %d of %d (%.1f%% of universe)" % (
    Cframe["account_key"].nunique(), frame["account_key"].nunique(),
    100*Cframe["account_key"].nunique()/frame["account_key"].nunique()))
print("  abuse as %% of ALL client winnings: %.1f%%" % (100*total_cost/max(total_client_win,1)))

# model accuracy from the (pre-trained) forecast
fc = rule_forecast.forecast(5)
accs = {p: d.get("holdout_auc") for p, d in (fc.get("profiles") or {}).items()}
print("\n=== PER-CLASS FORECAST ACCURACY (holdout AUC) ===")
for p, a in accs.items():
    print("  %-24s AUC=%s" % (p, a))
avg_auc = np.mean([a for a in accs.values() if a])
print("  mean AUC: %.3f" % avg_auc)

out = {"n_days": int(n_days), "horizon": N, "by_class": rows,
       "total_cost": total_cost, "cost_per_day": total_cost/n_days,
       "C_accounts": int(Cframe["account_key"].nunique()),
       "universe_accounts": int(frame["account_key"].nunique()),
       "abuse_share_of_winnings": float(total_cost/max(total_client_win,1)),
       "accuracies": accs, "mean_auc": float(avg_auc),
       "date_min": str(frame["decision_day"].min().date()),
       "date_max": str(frame["decision_day"].max().date())}
open(r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\taf_numbers.json","w").write(json.dumps(out, indent=2))
print("\nsaved taf_numbers.json")
