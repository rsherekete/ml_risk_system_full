import sys
from datetime import datetime, timedelta
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
from webapp import cashflow_store, rule_models

flows = cashflow_store.read_cashflows(
    start=datetime.utcnow() - timedelta(days=90))
amt = pd.to_numeric(flows["amount"], errors="coerce").fillna(0.0)
g = pd.DataFrame({"dep": amt.clip(lower=0), "wd": (-amt).clip(lower=0),
                  "account_key": flows["account_key"].astype(str)}
                 ).groupby("account_key").sum()
ratio = (g["wd"] / g["dep"].clip(lower=1e-9)).clip(0, 1)
ratio = ratio.where(g["dep"] > 0, 0.0)
corpus = rule_models._ad_latest()
cols = {}
for c in ("account_age_days", "life_pnl", "life_days_active",
          "life_win_rate", "life_profit_factor", "life_closes"):
    if c in corpus.columns:
        cols[c] = pd.to_numeric(corpus[c], errors="coerce")
joined = pd.DataFrame({"bx": ratio, "dep": g["dep"]}).join(
    pd.DataFrame(cols), how="inner")
print(f"accounts with flows+corpus: {len(joined):,}")
tests = [
    ("bx>=0.5 (old)", joined["bx"] >= 0.5),
    ("bx>=0.8 & age<=60", (joined["bx"] >= 0.8)
     & (joined["account_age_days"] <= 60)),
    ("bx>=0.9 & life_pnl>0 & life_days<=20",
     (joined["bx"] >= 0.9) & (joined["life_pnl"] > 0)
     & (joined["life_days_active"] <= 20)),
    ("bx>=0.9 & life_pnl>0 & life_days<=20 & pf>=1.5",
     (joined["bx"] >= 0.9) & (joined["life_pnl"] > 0)
     & (joined["life_days_active"] <= 20)
     & (joined["life_profit_factor"] >= 1.5)),
    ("bx>=0.95 & life_pnl>0 & life_days<=10 & pf>=2",
     (joined["bx"] >= 0.95) & (joined["life_pnl"] > 0)
     & (joined["life_days_active"] <= 10)
     & (joined["life_profit_factor"] >= 2.0)),
]
for name, m in tests:
    print(f"  {name:50s} -> {int(m.fillna(False).sum()):,}")
