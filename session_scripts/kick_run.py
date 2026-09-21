import sys, sqlite3, json, time, urllib.request
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth

db = r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db"
uid = sqlite3.connect(db).execute(
    "SELECT id FROM users WHERE role='admin' ORDER BY id LIMIT 1").fetchone()[0]
tok = auth.start_session(uid)   # DB-backed -> valid against the live server

def call(path, method="GET"):
    req = urllib.request.Request("http://127.0.0.1:8000" + path, method=method)
    req.add_header("Cookie", f"zfx_session={tok}")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

before = call("/api/quant/lab")
print("before: status =", before["status"])
kick = call("/api/quant/lab/run", "POST")
print("POST /api/quant/lab/run ->", kick)

# watch the first ~40s of the run stream in
for i in range(8):
    time.sleep(5)
    s = call("/api/quant/lab")
    print(f"  t+{(i+1)*5:2}s status={s['status']} progress={s['progress']:.3f} "
          f"live_pts={len(s['live'])} | {s['message'][:70]}")
    if s["status"] in ("done", "error"):
        break
print("\nlast log lines:")
for line in s["log"][-6:]:
    print("  ", line)
