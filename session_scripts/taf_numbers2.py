"""TAF abuse-cost -- conservative, account-level NET extraction (cent/contract
normalised). A = (account in C on >=1 day) AND (net realised P&L over period > 0).
Cost = net P&L of A accounts (what toxic-and-profitable accounts actually took)."""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from webapp import antifraud, rule_forecast

N = 5
frame = antifraud._frame().copy()
frame["decision_day"] = pd.to_datetime(frame["decision_day"])
pnl_col = next(c for c in ("realised_pnl","day_pnl","realized_pnl","pnl") if c in frame.columns)
frame["pnl"] = pd.to_numeric(frame[pnl_col], errors="coerce").fillna(0.0)
n_days = frame["decision_day"].dt.normalize().nunique()

# account-level net P&L over the whole window (client gain = broker/B-book loss)
acct_net = frame.groupby("account_key")["pnl"].sum()

masks = rule_forecast._masks(antifraud.load_rules())
class_members = {}          # class -> set of account_keys ever flagged
union_members = set()
for cls, (needed, build) in masks.items():
    if any(c not in frame.columns for c in needed):
        continue
    try:
        m = build(frame).fillna(False).to_numpy()
    except Exception:
        continue
    if m.mean() < 0.0005 or m.mean() > 0.6:
        continue
    accts = set(frame.loc[m, "account_key"].unique())
    class_members[cls] = accts
    union_members |= accts

def cost_of(accts):
    net = acct_net.reindex(list(accts)).fillna(0.0)
    winners = net[net > 0]
    return float(winners.sum()), int(len(winners)), int(len(accts))

rows = []
for cls, accts in class_members.items():
    cost, n_a, n_c = cost_of(accts)
    rows.append({"class": cls, "C_accounts": n_c, "A_accounts": n_a,
                 "A_share": round(n_a/max(n_c,1),3),
                 "cost_total": cost, "cost_per_day": cost/n_days})
byc = sorted(rows, key=lambda r: -r["cost_per_day"])
print("=== ABUSE COST BY CLASS (account-level net, ordered by daily impact) ===")
for r in byc:
    print("  %-24s C=%6d  A=%6d (%2.0f%%)  cost=$%13s  /day=$%10s" % (
        r["class"], r["C_accounts"], r["A_accounts"], r["A_share"]*100,
        f"{r['cost_total']:,.0f}", f"{r['cost_per_day']:,.0f}"))

u_cost, u_A, u_C = cost_of(union_members)
total_win = float(acct_net[acct_net > 0].sum())
uni_accts = frame["account_key"].nunique()
print("\n=== UNIVERSE (deduplicated union) ===")
print("  accounts: {:,} | in C: {:,} ({:.1f}%) | ABUSE (A): {:,}".format(
    uni_accts, u_C, 100*u_C/uni_accts, u_A))
print("  TOTAL ABUSE COST: ${:,.0f}  over {} days  =  ${:,.0f}/day".format(u_cost, n_days, u_cost/n_days))
print("  abuse as % of ALL client net winnings: {:.1f}%".format(100*u_cost/max(total_win,1)))

fc = rule_forecast.forecast(5)
accs = {p: d.get("holdout_auc") for p, d in (fc.get("profiles") or {}).items()}
avg_auc = float(np.mean([a for a in accs.values() if a]))
# recoverable @ target accuracy (recall proxy = AUC-implied catch rate, capped)
target_acc = 0.75
recoverable_day = u_cost/n_days * target_acc
print("\n=== FORECAST ACCURACY ===")
for p, a in accs.items(): print("  %-24s AUC=%s" % (p, a))
print("  mean AUC: {:.3f}".format(avg_auc))
print("\n=== SMART TARGET ===")
print("  target OOS accuracy: {:.0%} | 2-week live test".format(target_acc))
print("  recoverable at target: ${:,.0f}/day  (= {:.0%} of daily abuse cost)".format(
    recoverable_day, target_acc))

out = {"n_days": int(n_days), "horizon": N, "by_class": byc,
       "universe_accounts": int(uni_accts), "C_accounts": int(u_C), "A_accounts": int(u_A),
       "total_cost": u_cost, "cost_per_day": u_cost/n_days,
       "abuse_share_of_winnings": float(u_cost/max(total_win,1)),
       "accuracies": accs, "mean_auc": avg_auc, "target_acc": target_acc,
       "recoverable_per_day": recoverable_day,
       "date_min": str(frame["decision_day"].min().date()),
       "date_max": str(frame["decision_day"].max().date())}
open(r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\taf_numbers.json","w").write(json.dumps(out, indent=2))
print("\nsaved taf_numbers.json")
