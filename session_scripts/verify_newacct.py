import sys, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth, vantage as v
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}
base = "http://127.0.0.1:8000"
d = requests.get(base + "/api/vantage/status", cookies=cook, timeout=40).json()
e = d["engine"]
if not e.get("running"):
    print("engine not running yet; starting...")
    requests.post(base + "/vantage/start", cookies=cook, timeout=40); time.sleep(6)
    d = requests.get(base + "/api/vantage/status", cookies=cook, timeout=40).json(); e = d["engine"]
cfg = v.load_config()
print("engine running=%s last_error=%s" % (e.get("running"), e.get("last_error")))
try:
    snap = v.account_snapshot()
    print("CONNECTED ACCOUNT: login=%s server=%s" % (snap.get("login"), snap.get("server") or cfg.server))
    print("  balance=$%s equity=$%s leverage=1:%s currency=%s"
          % (snap.get("balance"), snap.get("equity"), snap.get("leverage"), snap.get("currency")))
except Exception as ex:
    print("snapshot err:", ex)
print("MODE(normal): fixed_lot=%.3f risk_budget=%.2f dd_stress=%.4f slots=%d max_positions=%d max_symbol_share=%.2f"
      % (cfg.fixed_lot, cfg.risk_budget_fraction, cfg.dd_stress_move, cfg.expected_peak_positions,
         cfg.max_open_positions, cfg.max_symbol_share))
print("feeds kafka=%s mysql=%s | scored=%s filled=%s | open positions=%d"
      % (d.get("feed", {}).get("kafka"), d.get("feed", {}).get("mysql"),
         e.get("scored"), e.get("filled"), len(d.get("report", {}).get("positions", []))))
