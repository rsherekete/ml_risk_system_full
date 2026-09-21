import sys, urllib.request, json, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
def get(p, t=180):
    req = urllib.request.Request("http://127.0.0.1:8000" + p)
    req.add_header("Cookie", "zfx_session=" + tok)
    return json.loads(urllib.request.urlopen(req, timeout=t).read())
t = time.time()
try:
    r = get("/api/antifraud/forecast?horizon=5")
    print("forecast in %.1fs:" % (time.time() - t))
    print("  keys:", list(r.keys())[:12])
    print("  error:", r.get("error"))
    print("  trained_minutes:", r.get("trained_minutes"), "| horizon_days:", r.get("horizon_days"))
    profs = r.get("profiles") or {}
    print("  profiles trained:", list(profs.keys()))
    for name, d in list(profs.items())[:4]:
        print("    %-20s AUC=%s conversions=%s watchlist=%d" % (
            name, d.get("holdout_auc"), d.get("conversions_trained_on"), len(d.get("watchlist") or [])))
    print("  skipped:", r.get("skipped"))
except Exception as e:
    print("forecast ERR after %.1fs:" % (time.time() - t), type(e).__name__, e)
