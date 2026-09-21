"""Run the 4 Sep NFP analysis IN THIS PROCESS (no server restart needed to
test the linked-account / rebate / notional build), write the workbook to
docs/ and print what changed."""
import sys, time, json
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 50)
from webapp import event_impact

p = event_impact.preset("nfp_2026_09_04")
print("preset:", p)
t0 = time.time()
data = event_impact._analyze_inner(p["start"], p["end"], p["symbols"], 500.0, 500.0, row_cap=None)
print(f"[{time.time()-t0:.0f}s] clients {data.get('n_impacted')} | accounts {data.get('n_accounts')} | linked {data.get('n_linked')}")
print("abuse:", data.get("abuse_counts")); print("classes:", data.get("classes")); print("segments:", data.get("segments"))
print("totals:", {k: v for k, v in (data.get("totals") or {}).items() if k in ("window_pnl", "notional_usd_total", "rebates_total", "net_revenue_total", "gross_revenue_total", "net_deposits_total", "median_mltv")})
for n in data.get("abuse_notes") or []:
    print("  note:", n)
rows = pd.DataFrame(data["rows"])
cols = ["account", "subaccounts", "accounts_in_window", "link_source", "client_class", "window_pnl", "net_deposits",
        "life_rebates", "equity", "net_revenue", "gross_revenue", "notional_usd", "notional_usd_monthly", "active_life_months", "mltv", "monthly_revenue"]
print("\n--- clients with linked accounts (top by |window pnl|)")
lk = rows[rows["subaccounts"] > 0].copy()
lk["abs"] = lk["window_pnl"].abs()
print(lk.sort_values("abs", ascending=False)[cols].head(15).to_string())
print("\n--- rebate receivers")
print(rows[rows["life_rebates"] > 0][cols].sort_values("life_rebates", ascending=False).head(10).to_string())
print("\n--- distribution: subaccounts", rows["subaccounts"].describe().to_dict())
print("accounts_in_window>1:", int((rows["accounts_in_window"] > 1).sum()), "| link_source:", rows["link_source"].value_counts().to_dict())
print("notional_usd: sum", f"{rows['notional_usd'].sum():,.0f}", "median", f"{rows['notional_usd'].median():,.0f}", "zero:", int((rows["notional_usd"] <= 0).sum()))
print("equity_source:", rows["equity_source"].value_counts().to_dict())
blob = event_impact.build_excel(data)
out = r"c:\Users\RoyVivasi\Documents\notebook\docs\event_impact_XAUUSD_2026-09-04_v3.xlsx"
open(out, "wb").write(blob)
print(f"\nexcel {len(blob):,} bytes -> {out}")
try:
    event_impact.LAST_PATH.write_text(json.dumps(data, default=str), encoding="utf-8")
    print("cached ->", event_impact.LAST_PATH)
except Exception as e:
    print("cache write failed:", e)
