import sys, sqlite3, time, os, urllib.request, json, datetime as dt
from collections import defaultdict
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth

since = time.time() - 2 * 3600      # last 2 hours
cx = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
cx.row_factory = sqlite3.Row

# EXPECTED + activity from the order log (both tables) -- FILLED trades last 2h
exp = defaultdict(lambda: {"n": 0, "expected": 0.0, "scores": []})
fills = []
for table in ("vantage_orders", "vantage_orders_archived"):
    try:
        for r in cx.execute(f"SELECT created, stance, symbol, our_lots, expected_usd, live_score "
                            f"FROM {table} WHERE created > ? AND status='filled'", (since,)):
            s = r["stance"] or "?"
            exp[s]["n"] += 1
            exp[s]["expected"] += float(r["expected_usd"] or 0)
            if r["live_score"] is not None: exp[s]["scores"].append(float(r["live_score"]))
            fills.append(dict(r))
    except Exception as e:
        print(table, "err", e)

print("=== LAST 2 HOURS: filled trades by stance (EXPECTED) ===")
for s, d in exp.items():
    sc = d["scores"]
    print("  %-8s n=%3d  expected_total=$%8.2f  avg_expected=$%.3f  avg_score=%.3f" % (
        s, d["n"], d["expected"], d["expected"]/d["n"] if d["n"] else 0,
        sum(sc)/len(sc) if sc else 0))
print("  total fills:", len(fills))

# ACTUAL P&L by stance -- account_reconcile via app, reset bypassed
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
RF = r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts\stats_reset.txt"
moved = os.path.exists(RF)
if moved: os.rename(RF, RF + ".bak")
def get(p):
    req = urllib.request.Request("http://127.0.0.1:8000" + p)
    req.add_header("Cookie", "zfx_session=" + tok)
    return json.loads(urllib.request.urlopen(req, timeout=90).read())
try:
    # reconcile over a short window to approximate 'recent'
    rec = get("/api/vantage/reconcile")
    print("\n=== ACTUAL realized by stance (recent window, reset bypassed) ===")
    print("  balance=$%.2f equity=$%.2f trading_realized=$%.2f deals=%s" % (
        rec.get("balance", 0), rec.get("equity", 0),
        rec.get("total_trading_realized", 0), rec.get("trade_deals")))
    for st, d in (rec.get("by_stance") or {}).items():
        print("  %-8s actual_pnl=$%.2f" % (st, d.get("pnl", 0)))
finally:
    if moved: os.rename(RF + ".bak", RF)
