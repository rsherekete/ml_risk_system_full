import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth
import webapp.vantage as v
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
d = requests.get("http://127.0.0.1:8000/api/vantage/status",
                 cookies={"zfx_session": tok}, timeout=40).json()
e = d["engine"]
print("engine running=%s mode=%s scored=%s acted=%s filled=%s blocked=%s"
      % (e.get("running"), e.get("mode"), e.get("scored"), e.get("acted"),
         e.get("filled"), e.get("blocked")))
rep = d["report"]
print("risk_budget_usd=%s  actual_floating_usd=%s  positions=%s"
      % (rep.get("risk_budget_usd"), rep.get("actual_floating_usd"),
         len(rep.get("positions", []))))
cfg = v.load_config()
print("config: max_gross_leverage=%s min_lot=%s max_symbol_share=%s max_lots_per_trade=%s"
      % (cfg.max_gross_leverage, cfg.min_lot, cfg.max_symbol_share, cfg.max_lots_per_trade))
cfg.leverage = 500.0
print("LIVE exposure wall = %.0f x equity (broker 500x capped to %.0f)"
      % (v._wall_lev(cfg), cfg.max_gross_leverage))
print("_trim_to_safe_margin removed:", not hasattr(v, "_trim_to_safe_margin"))
