import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
from webapp import cashflow_store

f = cashflow_store.read_cashflows()
print("rows:", len(f), "| cols:", list(f.columns))
if len(f):
    f["when"] = pd.to_datetime(f["when"], errors="coerce")
    print("date range:", f["when"].min(), "->", f["when"].max())
    amt = pd.to_numeric(f["amount"], errors="coerce")
    print("amount sign counts: neg", int((amt < 0).sum()),
          "| pos", int((amt > 0).sum()), "| zero", int((amt == 0).sum()))
    if "kind" in f.columns:
        print("kinds:", f["kind"].value_counts().head(8).to_dict())
    # coverage in the last 10 days
    recent = f[f["when"] >= pd.Timestamp("2026-09-01")]
    print("rows since Sep 1:", len(recent))
    if len(recent):
        print("  by day:", recent.groupby(recent["when"].dt.date).size().to_dict())
        rn = pd.to_numeric(recent["amount"], errors="coerce")
        print("  withdrawals (amount<0) since Sep 1:", int((rn < 0).sum()),
              "totaling", round(float(rn[rn < 0].sum()), 2))
