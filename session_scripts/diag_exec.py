import sys, time, collections
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth, vantage as v
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}; base = "http://127.0.0.1:8000"
d = requests.get(base + "/api/vantage/status", cookies=cook, timeout=40).json()
e = d["engine"]; snap = d.get("report", {}).get("snapshot", {}); cfg = v.load_config()
print("config: min_lot=%s copy_anchor=%s invert_anchor=%s" % (cfg.min_lot, cfg.copy_anchor_override, cfg.invert_anchor_override))
bal = float(snap.get("balance") or 0); eq = float(snap.get("equity") or 0); mg = float(snap.get("margin") or 0)
lvl = (eq/mg*100) if mg else 1e9
print("account: balance=%.2f equity=%.2f margin=%.2f level=%.0f%%" % (bal, eq, mg, lvl))
print("CIRCUIT BREAKER would trigger if: equity<%.1f (%.55f?) OR level<%s" % (bal*cfg.equity_floor_frac, eq, cfg.safe_margin_level))
paused = bal > 0 and (eq < bal*cfg.equity_floor_frac or lvl < cfg.safe_margin_level)
print("  => entries_paused likely = %s" % paused)
print("\nscored=%s acted=%s filled=%s blocked=%s stale=%s"
      % (e.get("scored"), e.get("acted"), e.get("filled"), e.get("blocked"), e.get("stale_skipped")))
# recent order-log entries (all statuses)
orders = v.recent_orders(200)
recent = [o for o in orders if (o.get("created") or 0) >= time.time() - 300]
print("order-log entries last 5min: %d (acted=%s -> gap means intents dropped before execute)" % (len(recent), e.get("acted")))
for k, n in collections.Counter((o.get("status"), o.get("stance")) for o in recent).most_common():
    print("  %-9s %-7s x%d" % (k[0], k[1], n))
# log clues
log = e.get("log") or []
for kw in ("MARGIN GUARD", "circuit", "pausing", "RECYCLED", "netted", "LIMIT PLACED", "entries resume"):
    hits = [l for l in log if kw.lower() in l.lower()]
    if hits:
        print("LOG[%s]: %s" % (kw, hits[-1][:90]))
print("\nlog tail:")
for l in log[-6:]:
    print("  ", l[:110])
