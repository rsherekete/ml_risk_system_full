import sys, json
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}; base = "http://127.0.0.1:8000"
d = requests.get(base + "/api/vantage/status", cookies=cook, timeout=40).json()
e = d["engine"]
print("engine keys:", sorted(e.keys()))
for k in ("running", "connected", "mode", "last_error", "account", "login", "balance",
          "equity", "leverage", "scored", "filled"):
    if k in e:
        print("  engine.%s = %r" % (k, e[k]))
rep = d.get("report", {})
snap = rep.get("snapshot")
print("\nreport.snapshot =", json.dumps(snap) if not isinstance(snap, dict) else json.dumps(snap, default=str)[:600])
# any log in engine
log = e.get("log") or e.get("recent_log") or e.get("messages")
if log:
    print("\nengine log tail:")
    for line in (log[-12:] if isinstance(log, list) else [log]):
        print("  ", line)
# try the diag endpoint
try:
    diag = requests.get(base + "/api/vantage/diag?hours=1", cookies=cook, timeout=40).json()
    print("\ndiag keys:", list(diag.keys())[:12])
except Exception as ex:
    print("diag err:", ex)
