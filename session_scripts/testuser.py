import sys, urllib.request, urllib.parse, http.cookiejar, json
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth

with auth.connect() as c:
    row = c.execute("SELECT id, username, role FROM users WHERE role != 'admin' LIMIT 1").fetchone()
    if row is None:
        print("no non-admin user exists"); sys.exit()
    print("non-admin user:", row["username"], "| role:", row["role"])
    tok = auth.start_session(row["id"])

def get(path):
    req = urllib.request.Request("http://127.0.0.1:8000" + path)
    req.add_header("Cookie", "zfx_session=" + tok)
    return urllib.request.urlopen(req, timeout=30).read().decode()

print("health:", get("/api/agent/health"))
