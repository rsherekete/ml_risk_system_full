import sys, sqlite3, json, urllib.request, http.cookiejar
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth

db = r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db"
row = sqlite3.connect(db).execute(
    "SELECT id, username, role FROM users WHERE role='admin' "
    "ORDER BY id LIMIT 1").fetchone()
print("admin user:", row)
tok = auth.start_session(row[0])

req = urllib.request.Request("http://127.0.0.1:8000/api/vantage/status")
req.add_header("Cookie", f"session={tok}")
try:
    j = json.loads(urllib.request.urlopen(req, timeout=25).read())
except Exception as e:
    print("request failed:", e); sys.exit()
if j.get("error"):
    print("ERROR:", j["error"]); sys.exit()
eng = j.get("engine", {})
print("engine.running:", eng.get("running"),
      "| strategy1:", eng.get("strategy1"),
      "| entries_paused:", j.get("entries_paused"))
for k in ("scored", "acted", "filled", "blocked", "stale_skipped"):
    print(f"  {k}: {j.get(k)}")
print("stream:", json.dumps(j.get("stream"), default=str)[:400])
rep = j.get("report", {})
if isinstance(rep, dict):
    print("snapshot:", json.dumps(rep.get("snapshot"), default=str)[:200])
print("signals count:", len(j.get("signals", [])))
for s in (j.get("signals") or [])[:4]:
    print("  ", {k: s.get(k) for k in
                 ("symbol", "stance", "score", "traded", "route")})
