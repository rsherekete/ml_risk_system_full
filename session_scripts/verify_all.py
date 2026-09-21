import sys, time, collections
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth, vantage as v
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}; base = "http://127.0.0.1:8000"
d = requests.get(base + "/api/vantage/status", cookies=cook, timeout=40).json()
e = d["engine"]
print("running=%s connected=%s scored=%s acted=%s filled=%s blocked=%s stale=%s"
      % (e.get("running"), e.get("connected"), e.get("scored"), e.get("acted"),
         e.get("filled"), e.get("blocked"), e.get("stale_skipped")))
snap = d.get("report", {}).get("snapshot", {})
print("account: login=%s balance=%s equity=%s currency=%s leverage=%s | open positions=%d"
      % (snap.get("login"), snap.get("balance"), snap.get("equity"), snap.get("currency"),
         snap.get("leverage"), len(d.get("report", {}).get("positions", []))))
# engine log: market watch, connect, duckdb errors
log = e.get("log") or []
mw = [l for l in log if "market watch" in l.lower()]
dberr = [l for l in log if "store busy" in l.lower() or "duckdb" in l.lower()]
sym = [l for l in log if "symbol map" in l.lower()]
print("\nMARKET WATCH log:", mw[-1] if mw else "(none yet)")
print("SYMBOL MAP log:  ", sym[-1] if sym else "(none)")
print("stream/duckdb errors in log: %d (last: %s)" % (len(dberr), dberr[-1][:80] if dberr else "none"))
print("\nlog tail:")
for l in log[-6:]:
    print("  ", l[:120])
# recent signals + block reasons
print("\nrecent stream signals:")
for s in d.get("signals", [])[:6]:
    print("  %s dec=%s lots=%s traded=%s" % (s.get("symbol"), s.get("decision"), s.get("lots"), s.get("traded")))
orders = v.recent_orders(150); recent = [o for o in orders if (o.get("created") or 0) >= time.time() - 300]
print("\norders last 5min: %d" % len(recent))
for k, n in collections.Counter((o.get("status"), o.get("stance"), o.get("symbol"), o.get("our_lots"), (o.get("detail") or "")[:30]) for o in recent).most_common(12):
    print("  %-8s %-7s %-10s lots=%-5s %s x%d" % (k[0], k[1], k[2], k[3], k[4], n))
