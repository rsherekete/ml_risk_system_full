import sys, urllib.request, json, os, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth

with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])

RF = r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts\stats_reset.txt"
BAK = RF + ".bak"
moved = False
if os.path.exists(RF):
    os.rename(RF, BAK); moved = True

def get(path):
    req = urllib.request.Request("http://127.0.0.1:8000" + path)
    req.add_header("Cookie", "zfx_session=" + tok)
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

try:
    rec = get("/api/vantage/reconcile")
    print("FULL reconcile (reset bypassed):")
    print("  balance=$%.2f equity=$%.2f" % (rec.get("balance", 0), rec.get("equity", 0)))
    print("  total_trading_realized=$%.2f | trade_deals=%s" % (
        rec.get("total_trading_realized", 0), rec.get("trade_deals")))
    print("  deposits_withdrawals=$%.2f | balance_deals=%s" % (
        rec.get("deposits_withdrawals", 0), rec.get("balance_deals")))
    print("  by_stance:")
    for st, d in (rec.get("by_stance") or {}).items():
        print("    %-10s pnl=$%.2f n=%s" % (st, d.get("pnl", 0), d.get("n")))
finally:
    if moved:
        os.rename(BAK, RF)
        print("\n(reset file restored)")
