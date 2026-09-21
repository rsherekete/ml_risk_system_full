"""Through the live server: does the Event Impact API serve the client-level
fields, and does the Excel route still build?"""
import sys, sqlite3, json, urllib.request, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth
uid = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db").execute(
    "SELECT id FROM users WHERE role='admin' ORDER BY id LIMIT 1").fetchone()[0]
tok = auth.start_session(uid)

def get(path, timeout=300):
    req = urllib.request.Request("http://127.0.0.1:8000" + path); req.add_header("Cookie", f"zfx_session={tok}")
    return urllib.request.urlopen(req, timeout=timeout).read()

t0 = time.time()
d = json.loads(get("/api/antifraud/event_impact"))          # cached last analysis
print(f"[{time.time()-t0:.1f}s] keys: n_impacted={d.get('n_impacted')} n_accounts={d.get('n_accounts')} n_linked={d.get('n_linked')} rows={len(d.get('rows', []))}")
r = next((x for x in d.get("rows", []) if x.get("subaccounts")), None)
print("sample linked row:", {k: r.get(k) for k in ("account", "subaccounts", "accounts_in_window", "link_source", "client_class",
                                                  "window_pnl", "net_revenue", "gross_revenue", "notional_usd_monthly", "life_rebates")} if r else None)
print("totals:", {k: d["totals"].get(k) for k in ("notional_usd_total", "rebates_total", "net_revenue_total")})
t0 = time.time()
blob = get("/api/antifraud/event_impact.xlsx")
print(f"[{time.time()-t0:.1f}s] xlsx {len(blob):,} bytes")
t0 = time.time()
page = get("/trading/antifraud").decode("utf-8", "replace")
print(f"[{time.time()-t0:.1f}s] antifraud page {len(page):,} chars | has Sub-acc column: {'Sub-acc' in page} | Net revenue: {'Net revenue' in page}")
