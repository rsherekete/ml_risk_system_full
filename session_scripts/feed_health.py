import sys, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}; base = "http://127.0.0.1:8000"
def snap():
    d = requests.get(base + "/api/vantage/status", cookies=cook, timeout=40).json()
    e = d["engine"]
    hb = [l for l in (e.get("log") or []) if "heartbeat" in l]
    return e.get("scored"), e.get("acted"), e.get("filled"), (hb[-1] if hb else "")
s1, a1, f1, hb1 = snap()
print("[t0]   scored=%s acted=%s filled=%s" % (s1, a1, f1))
print("       %s" % hb1[:120])
time.sleep(60)
s2, a2, f2, hb2 = snap()
print("[t+60] scored=%s acted=%s filled=%s" % (s2, a2, f2))
print("       %s" % hb2[:120])
print("\nscored/min ~ %d | acted +%d | filled +%d" % (s2 - s1, a2 - a1, f2 - f1))
