import sys, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth, vantage as v
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}; base = "http://127.0.0.1:8000"
print("=== recent BTC block/reject details (full string) ===")
for o in v.recent_orders(60):
    if (o.get("created") or 0) >= time.time() - 900 and o.get("status") in ("blocked", "rejected"):
        print("  [%s] %s %s: %s" % (o.get("status"), o.get("stance"), o.get("symbol"), o.get("detail")))

def snap():
    d = requests.get(base + "/api/vantage/status", cookies=cook, timeout=40).json()
    e = d["engine"]; sg = d.get("signals", [])
    top = (sg[0].get("symbol"), sg[0].get("decision"), sg[0].get("traded")) if sg else None
    return e.get("scored"), len(sg), top, (e.get("log") or [""])[-1][:85]
s1, n1, t1, l1 = snap()
print("\n[t0]    scored=%s signals=%d newest=%s" % (s1, n1, t1))
print("        log: %s" % l1)
time.sleep(22)
s2, n2, t2, l2 = snap()
print("[t+22s] scored=%s signals=%d newest=%s" % (s2, n2, t2))
print("        log: %s" % l2)
print("\nscoring advancing: %s | newest signal changed: %s" % (s2 != s1, t2 != t1))
