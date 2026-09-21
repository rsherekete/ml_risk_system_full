import sys, time, collections
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth, vantage as v
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
d = requests.get("http://127.0.0.1:8000/api/vantage/status", cookies={"zfx_session": tok}, timeout=40).json()
e = d["engine"]; feed = d.get("feed", {}); cfg = v.load_config()
print("engine running=%s feeds k=%s m=%s | scored=%s acted=%s filled=%s blocked=%s"
      % (e.get("running"), feed.get("kafka"), feed.get("mysql"), e.get("scored"),
         e.get("acted"), e.get("filled"), e.get("blocked")))
print("MODE: fixed_lot=%.3f  strategy1=%s strategy2=%s  max_open_positions=%d  max_symbol_share=%.2f"
      % (cfg.fixed_lot, cfg.strategy1, cfg.strategy2, cfg.max_open_positions, cfg.max_symbol_share))
snap = d.get("report", {})
print("open positions=%d floating=%s" % (len(snap.get("positions", [])), snap.get("actual_floating_usd")))
rows = v.recent_orders(300)
recent = [r for r in rows if (r.get("created") or 0) >= time.time() - 240]
print("\norders last 4min: %d" % len(recent))
for k, n in collections.Counter((r.get("status"), r.get("stance"), r.get("our_lots")) for r in recent).most_common(10):
    print("  status=%-9s stance=%-7s lots=%s  x%d" % (k[0], k[1], k[2], n))
