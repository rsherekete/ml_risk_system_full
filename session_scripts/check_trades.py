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
print("scored=%s acted=%s filled=%s blocked=%s stale=%s | positions=%d"
      % (e.get("scored"), e.get("acted"), e.get("filled"), e.get("blocked"),
         e.get("stale_skipped"), len(d.get("report", {}).get("positions", []))))
# store-busy errors WITH timestamps -> are any NEW (recent)?
busy = [l for l in (e.get("log") or []) if "store busy" in l.lower()]
print("store-busy log entries: %d ; newest: %s" % (len(busy), (busy[-1][:14] if busy else "none")))
print("newest log line: %s" % ((e.get("log") or [])[-1][:90] if e.get("log") else ""))
# recent order detail
orders = v.recent_orders(200)
recent = [o for o in orders if (o.get("created") or 0) >= time.time() - 420]
print("\norders last 7min: %d" % len(recent))
for k, n in collections.Counter((o.get("status"), o.get("stance"), o.get("symbol"), (o.get("detail") or "")[:34]) for o in recent).most_common(14):
    print("  %-8s %-7s %-11s %s x%d" % (k[0], k[1], k[2], k[3], n))
# any drawdown-budget blocks left?
ddblk = [o for o in recent if "drawdown budget" in (o.get("detail") or "").lower()]
print("\ndrawdown-budget blocks (should be ZERO): %d" % len(ddblk))
fills = [o for o in recent if o.get("status") == "filled"]
print("fills last 7min: %d" % len(fills))
for o in fills[:6]:
    print("  FILLED %s %s %s lots @ %s" % (o.get("stance"), o.get("symbol"), o.get("our_lots"), o.get("fill_price")))
