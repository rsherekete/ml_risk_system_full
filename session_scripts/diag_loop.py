import sys, time, datetime
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
    return e.get("scored"), e.get("acted"), e.get("stale_skipped"), (e.get("log") or [])
s1, a1, st1, log = snap()
print("server clock now ~", datetime.datetime.now().strftime("%H:%M:%S"))
print("[t0] scored=%s acted=%s stale=%s" % (s1, a1, st1))
print("\nlast 14 log lines:")
for l in log[-14:]:
    print("  ", l[:110])
print("\n...waiting 70s to see if the loop advances...")
time.sleep(70)
s2, a2, st2, log2 = snap()
print("[t+70] scored=%s acted=%s stale=%s" % (s2, a2, st2))
print("newest log line: %s" % (log2[-1][:110] if log2 else ""))
print("\nLOOP ALIVE (scored or stale advanced, or new log): scored+%d stale+%d newlog=%s"
      % (s2 - s1, st2 - st1, log2[-1] != log[-1]))
