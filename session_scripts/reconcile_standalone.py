"""Reconcile the standalone script's 4 Sep result against the application's own
analysis, run on the same footing: MySQL servers only (the app's store also
holds mt5_dubai_live01 from BigQuery), cash movements pinned to the same
as-of timestamp, and the same behavioural labels."""
import sys, json
import pandas as pd
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import data_store, event_impact as ei

MYSQL = ("mt4_live01", "mt4_live02", "mt4_live03", "mt4_live04", "mt5_live01")
_orig = data_store.read_history
data_store.read_history = lambda *a, **k: _orig(databases=MYSQL, **{kk: v for kk, v in k.items() if kk != "databases"}) \
    if not a else _orig(*a, **k)

app = ei._analyze_inner("2026-09-04 13:29:00", "2026-09-04 13:31:00", "XAUUSD", 500.0, 500.0)
sa = json.load(open(r"c:\Users\RoyVivasi\Documents\notebook\docs\replication\summary.json"))

print("=== totals: application (MySQL servers only) vs standalone ===")
print(f"{'metric':22} {'app':>16} {'standalone':>16} {'diff':>12}")
def row(k, a, b):
    d = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else ""
    print(f"{k:22} {a:>16,} {b:>16,} {d:>12,}" if isinstance(a, (int, float)) else f"{k:22} {str(a):>16} {str(b):>16}")
row("n_impacted", app["n_impacted"], sa["n_impacted"])
for k in ("window_pnl", "opens_in_window", "closes_in_window", "held_through", "actions_in_window",
          "lots_in_window", "stopped_out", "significant_loss", "significant_profit",
          "deposited_5d", "withdrawn_5d", "net_deposit_5d", "net_deposits_total", "median_mltv", "behaviour_changed"):
    row(k, app["totals"][k], sa["totals"][k])
print("\n=== segments ===")
for s in ("abuse_candidate", "retain_high_value", "compensate_review", "monitor", "minimal"):
    row(s, app["segments"].get(s, 0), sa["segments"].get(s, 0))
print("\napp cashflow_current_to:", app["cashflow_current_to"], "| standalone:", sa["cashflow_current_to"])
print("app rows (capped):", len(app["rows"]), "| standalone rows_in_workbook:", sa.get("rows_in_workbook"))
