import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
from webapp import model_service as ms
from webapp.trade_feed import cent_logins

tr = ms.load_scores(ms.VIEW_TRADING).copy()
qt = ms.load_scores(ms.VIEW_QUANT).copy()
tr["day"] = pd.to_datetime(tr["day"]).dt.normalize()
qt["day"] = pd.to_datetime(qt["day"]).dt.normalize()
lo = max(tr["day"].min(), qt["day"].min()); hi = min(tr["day"].max(), qt["day"].max())
tr = tr[(tr["day"] >= lo) & (tr["day"] <= hi)]
qt = qt[(qt["day"] >= lo) & (qt["day"] <= hi)]

ta = pd.to_numeric(tr["pnl"], errors="coerce").groupby(tr["account_key"]).sum()
qa = pd.to_numeric(qt["pnl"], errors="coerce").groupby(qt["account_key"]).sum()
j = pd.DataFrame({"trading": ta, "quant": qa}).dropna()
j = j[(j["quant"].abs() > 50)]
j["ratio"] = j["trading"] / j["quant"]

# cent membership
cents = {}
for server in ("mt4_live01","mt4_live02","mt4_live03","mt4_live04","mt5_live01"):
    try: cents[server] = cent_logins(server)
    except Exception: cents[server] = set()
def is_cent(key):
    server, login = key.split(":", 1)
    return int(login) in cents.get(server, set())
j["cent"] = [is_cent(a) for a in j.index]

print(f"accounts compared: {len(j):,} | cent among them: {j['cent'].sum():,}")
print("\nmedian trading/quant ratio:")
print("  cent accounts    :", round(j.loc[j['cent'], 'ratio'].median(), 2))
print("  non-cent accounts:", round(j.loc[~j['cent'], 'ratio'].median(), 2))
print("\ntop discrepancy accounts:")
worst = j.assign(diff=(j['trading']-j['quant']).abs()).nlargest(8, 'diff')
for a, r in worst.iterrows():
    print(f"  {a:24s} cent={r['cent']} trading={r['trading']:>14,.0f} quant={r['quant']:>12,.0f} ratio={r['ratio']:.1f}")
print("\nsum split:")
print("  cent trading sum:", f"{j.loc[j['cent'],'trading'].sum():,.0f}",
      "| cent quant sum:", f"{j.loc[j['cent'],'quant'].sum():,.0f}")
print("  non-cent trading:", f"{j.loc[~j['cent'],'trading'].sum():,.0f}",
      "| non-cent quant :", f"{j.loc[~j['cent'],'quant'].sum():,.0f}")
