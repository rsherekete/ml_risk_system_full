import sys, time, collections
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth, vantage as v
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}; base = "http://127.0.0.1:8000"

print("toggling KAFKA feed OFF (MySQL stays the decision feed)...")
r = requests.post(base + "/vantage/feed/toggle", cookies=cook,
                  data={"source": "kafka", "on": "0"}, timeout=60, allow_redirects=False)
print("  toggle status:", r.status_code)
time.sleep(12)   # let the engine thread restart + reconnect

d = requests.get(base + "/api/vantage/status", cookies=cook, timeout=40).json()
e = d["engine"]; feed = d.get("feed", {})
print("\nfeeds: mysql=%s kafka=%s | running=%s connected=%s"
      % (feed.get("mysql"), feed.get("kafka"), e.get("running"), e.get("connected")))
print("scored=%s acted=%s filled=%s blocked=%s stale=%s"
      % (e.get("scored"), e.get("acted"), e.get("filled"), e.get("blocked"), e.get("stale_skipped")))
dberr = [l for l in (e.get("log") or []) if "store busy" in l.lower()]
print("duckdb 'store busy' errors still in recent log: %d" % len(dberr))
print("\nlog tail:")
for l in (e.get("log") or [])[-6:]:
    print("  ", l[:120])
