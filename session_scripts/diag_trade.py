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
sigs = d.get("signals", [])
print("\nstream signals shown: %d" % len(sigs))
for s in sigs[:6]:
    print("  %s %s dir=%s lots=%s score=%s decision=%s traded=%s" % (
        s.get("account_key"), s.get("symbol"), s.get("direction"), s.get("lots"),
        round(s.get("value") or 0, 3) if s.get("value") is not None else None,
        s.get("decision"), s.get("traded")))
# recent orders since connect (block reasons)
orders = v.recent_orders(120)
recent = [o for o in orders if (o.get("created") or 0) >= time.time() - 600]
print("\norders last 10min: %d" % len(recent))
for k, n in collections.Counter((o.get("status"), (o.get("detail") or "")[:44]) for o in recent).most_common(12):
    print("  %-9s %-46s x%d" % (k[0], k[1], n))
# engine log tail
for line in (e.get("log") or [])[-10:]:
    print("LOG:", line)
