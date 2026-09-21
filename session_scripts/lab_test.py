import sys, sqlite3, json, urllib.request
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth

db = r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db"
row = sqlite3.connect(db).execute(
    "SELECT id, username FROM users WHERE role='admin' ORDER BY id LIMIT 1").fetchone()
tok = auth.start_session(row[0])

def get(path, method="GET"):
    req = urllib.request.Request("http://127.0.0.1:8000" + path, method=method)
    req.add_header("Cookie", f"session={tok}")
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, r.read()
    except Exception as e:
        return "ERR", str(e).encode()

# 1) the JSON endpoint
st, body = get("/api/quant/lab")
print("GET /api/quant/lab ->", st)
try:
    j = json.loads(body)
    print("  keys:", sorted(j.keys()))
    print("  status:", j.get("status"), "| persisted pts:", len(j.get("persisted") or []),
          "| live pts:", len(j.get("live") or []), "| history_days:", j.get("history_days"),
          "| roc_auc:", j.get("roc_auc"))
    pc = j.get("persisted") or []
    if pc:
        print("  first persisted:", pc[0])
        print("  last persisted:", pc[-1])
except Exception as e:
    print("  not JSON:", body[:200])

# 2) the page renders (contains our tab markers)
st, body = get("/quant/lab")
print("GET /quant/lab ->", st, "| bytes:", len(body))
html = body.decode("utf-8", "ignore")
for marker in ("Strategy Lab", "labCum", "api/quant/lab", "Run walk-forward"):
    print(f"  contains '{marker}':", marker in html)
