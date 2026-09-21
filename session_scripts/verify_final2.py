import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth, vantage as v
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
d = requests.get("http://127.0.0.1:8000/api/vantage/status", cookies={"zfx_session": tok}, timeout=40).json()
e = d["engine"]; feed = d.get("feed", {})
cfg = v.load_config()
print("engine running=%s feeds kafka=%s mysql=%s | scored=%s acted=%s filled=%s blocked=%s"
      % (e.get("running"), feed.get("kafka"), feed.get("mysql"), e.get("scored"),
         e.get("acted"), e.get("filled"), e.get("blocked")))
print("sizer: risk_budget=%.2f dd_stress=%.4f slots=%d min_lot=%.3f"
      % (cfg.risk_budget_fraction, cfg.dd_stress_move, cfg.expected_peak_positions, cfg.min_lot))
pl = v._lot_economics("XAUUSD+", 4400.0)[1]
min_dep = cfg.min_lot * cfg.dd_stress_move * cfg.expected_peak_positions * pl / cfg.risk_budget_fraction
print("MIN DEPOSIT (gold) = $%.0f" % min_dep)
snap = d.get("report", {})
print("positions=%d actual_floating=%s" % (len(snap.get("positions", [])), snap.get("actual_floating_usd")))
for eq in (970, 1000, 1006, 1100):
    lot = v._size_position(cfg, "XAUUSD+", 4400.0, +1, [], eq, cfg.leverage)
    print("  eq=$%-5d -> per-gold-trade lot=%.4f  => %s"
          % (eq, lot, ("trades at %.2f" % (int(lot*100)/100)) if lot >= cfg.min_lot else "BLOCKED"))
