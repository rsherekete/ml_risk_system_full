import sys, sqlite3
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")

from starlette.testclient import TestClient
from webapp import auth
from webapp.main import app          # import registers routes; no startup run

db = r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db"
uid, uname = sqlite3.connect(db).execute(
    "SELECT id, username FROM users WHERE role='admin' ORDER BY id LIMIT 1").fetchone()
tok = auth.start_session(uid)        # same process -> in-memory session is seen

client = TestClient(app)             # no 'with' => lifespan/startup not triggered
client.cookies.set("zfx_session", tok)

r = client.get("/api/quant/lab")
print("GET /api/quant/lab ->", r.status_code)
j = r.json()
print("  keys:", sorted(j.keys()))
print("  status:", j.get("status"), "| persisted:", len(j.get("persisted") or []),
      "| live:", len(j.get("live") or []), "| history_days:", j.get("history_days"),
      "| roc_auc:", j.get("roc_auc"))

r = client.get("/quant/lab")
print("GET /quant/lab ->", r.status_code, "| bytes:", len(r.text))
for m in ("Strategy Lab", "labCum", "/api/quant/lab", "Run walk-forward",
          "Per-class models", "MAE drawdown"):
    print(f"  contains '{m}':", m in r.text)
