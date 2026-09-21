"""Per-account diff of the app's analysis vs the standalone for the 4 Sep event:
which accounts land in different segments, and which rule input differs."""
import sys, yaml
from datetime import datetime
import pandas as pd
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook"); sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook\docs")
import event_impact_standalone as sa
from webapp import data_store, event_impact as ei

MYSQL = ("mt4_live01", "mt4_live02", "mt4_live03", "mt4_live04", "mt5_live01")
_orig = data_store.read_history
data_store.read_history = lambda *a, **k: _orig(databases=MYSQL, **{kk: v for kk, v in k.items() if kk != "databases"})

cfg = yaml.safe_load(open(r"c:\Users\RoyVivasi\Documents\notebook\docs\replication\_servers_local.yaml"))
lab = pd.read_csv(r"c:\Users\RoyVivasi\Documents\notebook\docs\event_impact_labels.csv", dtype=str).fillna("")
labels = dict(zip(lab["account_key"], lab["profile"]))
as_of = datetime(2026, 9, 13, 14, 24, 21)
sa.ROW_CAP = 10**9
S = pd.DataFrame(sa.analyze(cfg, "2026-09-04 13:29:00", "2026-09-04 13:31:00", "XAUUSD", 500.0, 500.0, labels, as_of, lambda m: None)["rows"]).set_index("account")
A = pd.DataFrame(ei._analyze_inner("2026-09-04 13:29:00", "2026-09-04 13:31:00", "XAUUSD", 500.0, 500.0, row_cap=None)["rows"]).set_index("account")
print("rows app", len(A), "| standalone", len(S), "| common", len(A.index.intersection(S.index)))
J = A.join(S, lsuffix="_app", rsuffix="_sa", how="inner")
diff = J[J["segment_app"] != J["segment_sa"]]
print("segment mismatches:", len(diff))
print(diff.groupby(["segment_app", "segment_sa"]).size().to_string())
for side in ("app", "sa"):
    src = A if side == "app" else S
    print(f"\n{side}: P80 mltv {src['mltv'].quantile(0.8):,.0f} | P80 value_volume {src['value_volume'].quantile(0.8):,.2f} | high_value {int(src['high_value'].sum())}")
cols = ["impact", "window_pnl", "wd_d5", "mltv", "net_deposits", "tenure_months", "value_volume", "high_value", "base_trades_pd"]
pd.set_option("display.width", 250)
print("\nfirst 12 mismatches (app vs standalone):")
print(diff[[c + "_app" for c in cols] + [c + "_sa" for c in cols]].head(12).round(2).to_string())
# which single input explains most flips?
for c in ("high_value", "wd_d5", "mltv", "value_volume", "impact"):
    a, s = diff[c + "_app"], diff[c + "_sa"]
    print(f"{c:14} differs on {(a != s).sum()} of {len(diff)} mismatched accounts")
