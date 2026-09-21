import sys, json, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
from webapp import model_service as ms

sp, mp = ms.artifact_paths(ms.VIEW_TRADING)
meta = json.loads(open(mp, encoding="utf-8").read())
print("trading artifact trained_at:", time.ctime(meta["trained_at"]), "| rows", meta["rows"])
print("use_exposure_days:", meta["config"].get("use_exposure_days"))
tr = pd.read_parquet(sp)
print("trading cols:", sorted(tr.columns)[:22])
pnlcol = next((c for c in ("pnl", "day_pnl", "client_pnl", "net_profit") if c in tr.columns), None)
print("trading pnl col:", pnlcol)
if pnlcol:
    print("trading pnl sum:", "$%.0f" % pd.to_numeric(tr[pnlcol], errors="coerce").sum())
    if "account_key" in tr.columns:
        print("trading accounts:", tr["account_key"].nunique())
