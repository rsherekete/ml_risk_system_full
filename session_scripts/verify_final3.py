import sys, time, collections
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth, vantage as v
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}; base = "http://127.0.0.1:8000"
cfg = v.load_config()
print("CONFIG: copy>=%.2f invert<=%.2f min_lot=%.2f" % (cfg.copy_anchor_override, cfg.invert_anchor_override, cfg.min_lot))
d = requests.get(base + "/api/vantage/status", cookies=cook, timeout=40).json()
e = d["engine"]; snap = d.get("report", {}).get("snapshot", {})
print("account: equity=%.2f margin_level=%s | running=%s connected=%s"
      % (float(snap.get("equity") or 0),
         round(float(snap.get("equity") or 0)/float(snap.get("margin") or 1)*100) if snap.get("margin") else "n/a",
         e.get("running"), e.get("connected")))
print("scored=%s acted=%s filled=%s blocked=%s stale=%s" % (e.get("scored"), e.get("acted"), e.get("filled"), e.get("blocked"), e.get("stale")))
# stream: any copy/invert now?
dec = collections.Counter(s.get("decision") for s in d.get("signals", []))
top = sorted([s.get("score") for s in d.get("signals", []) if s.get("score") is not None], reverse=True)[:5]
print("stream decisions:", dict(dec), "| top scores:", top)
# ENGINE orders (stance set), with lots -> confirm 0.1 floor
orders = v.recent_orders(200)
recent = [o for o in orders if (o.get("created") or 0) >= time.time() - 400 and o.get("stance") in ("copy", "invert")]
print("\nENGINE copy/invert orders last ~7min: %d" % len(recent))
for o in recent[:12]:
    print("  [%s] %s %s our_lots=%s client_lots=%s %s" % (o.get("status"), o.get("stance"), o.get("symbol"),
          o.get("our_lots"), o.get("client_lots"), (o.get("detail") or "")[:34]))
# positions with engine comments
print("\npositions:")
for p in d.get("report", {}).get("positions", []):
    print("  %s dir=%s lots=%s comment=%r" % (p.get("symbol"), p.get("direction"), p.get("lots"), p.get("comment")))
