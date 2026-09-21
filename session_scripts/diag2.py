import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd, numpy as np
from webapp import trade_feed
from webapp.trade_features import _AD_DIR
from webapp import data_store

cents = trade_feed.cent_logins("mt5_live01")

# AD corpus: find money columns, check cent vs non-cent
ad = pd.read_parquet(_AD_DIR / "model_frame.parquet")
money_like = [c for c in ad.columns if any(k in c.lower() for k in
              ("pnl", "profit", "notional", "volume", "abs_pnl", "deposit", "equity"))][:6]
print("AD money-like cols:", money_like)
sub = ad[ad["account_key"].astype(str).str.startswith("mt5_live01:")].copy()
sub["login"] = pd.to_numeric(sub["account_key"].str.split(":").str[1], errors="coerce")
sub["cent"] = sub["login"].isin(cents)
for c in money_like:
    cc = sub.loc[sub["cent"], c].abs().mean(); nc = sub.loc[~sub["cent"], c].abs().mean()
    ratio = (cc / nc) if nc else float("nan")
    print("  %-22s cent=%.3f non-cent=%.3f ratio=%.1f%s" % (
        c, cc, nc, ratio, "  <-- INFLATED ~100x" if ratio > 20 else ""))

# Warehouse: check one recent month parquet for mt5_live01
print("\nWAREHOUSE (data_store) mt5_live01 recent:")
try:
    wh = data_store.read_history(databases=("mt5_live01",),
                                 columns=["login", "net_profit", "volume_lots"])
    wh["cent"] = pd.to_numeric(wh["login"], errors="coerce").isin(cents)
    cc = wh.loc[wh["cent"], "net_profit"].abs().mean(); nc = wh.loc[~wh["cent"], "net_profit"].abs().mean()
    lc = wh.loc[wh["cent"], "volume_lots"].mean(); ln = wh.loc[~wh["cent"], "volume_lots"].mean()
    print("  rows=%d | cent |pnl|=%.1f non-cent=%.1f (ratio %.1f)" % (len(wh), cc, nc, (cc/nc) if nc else 0))
    print("  cent lots=%.4f non-cent lots=%.4f (ratio %.1f)  %s" % (
        lc, ln, (lc/ln) if ln else 0, "<-- INFLATED" if (lc/ln if ln else 0) > 5 else "(ok)"))
except Exception as e:
    print("  warehouse read err:", e)
