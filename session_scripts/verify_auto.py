import sys, time, collections
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth, vantage as v
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}; base = "http://127.0.0.1:8000"
d = requests.get(base + "/api/vantage/status", cookies=cook, timeout=40).json()
e = d["engine"]; feed = d.get("feed", {})
print("running=%s connected=%s feeds kafka=%s mysql=%s" % (e.get("running"), e.get("connected"), feed.get("kafka"), feed.get("mysql")))
print("scored=%s acted=%s filled=%s blocked=%s stale=%s" % (e.get("scored"), e.get("acted"), e.get("filled"), e.get("blocked"), e.get("stale_skipped")))
log = e.get("log") or []
busy_new = [l for l in log if "store busy" in l.lower()]
kafka_err = [l for l in log if "kafka feed" in l.lower()]
print("store-busy in log: %d (newest %s) | kafka-feed soft errors: %d"
      % (len(busy_new), busy_new[-1][:8] if busy_new else "-", len(kafka_err)))
# ENGINE-placed orders since restart (stance set = engine, not manual)
orders = v.recent_orders(200)
recent = [o for o in orders if (o.get("created") or 0) >= time.time() - 300]
eng = [o for o in recent if o.get("stance") in ("copy", "invert", "fade15")]
print("\nENGINE orders last 5min: %d" % len(eng))
for k, n in collections.Counter((o.get("status"), o.get("stance"), o.get("symbol"), (o.get("detail") or "")[:30]) for o in eng).most_common(12):
    print("  %-8s %-7s %-11s %s x%d" % (k[0], k[1], k[2], k[3], n))
# positions with engine comments (encoded) vs manual (empty)
rep = d.get("report", {})
print("\nOPEN POSITIONS:")
for p in rep.get("positions", []):
    tag = "ENGINE" if (p.get("comment") or "") else "manual/empty"
    print("  %s %s lots=%s comment=%r [%s]" % (p.get("symbol"), p.get("direction"), p.get("lots"), p.get("comment"), tag))
print("\nlog tail:")
for l in log[-6:]:
    print("  ", l[:110])
