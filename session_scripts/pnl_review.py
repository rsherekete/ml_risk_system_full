import sys, urllib.request, json
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
def get(path, t=90):
    req = urllib.request.Request("http://127.0.0.1:8000" + path)
    req.add_header("Cookie", "zfx_session=" + tok)
    return json.loads(urllib.request.urlopen(req, timeout=t).read())

print("=== account_reconcile (ground truth P&L by stance, since reset) ===")
rec = get("/api/vantage/reconcile")
print(json.dumps(rec, indent=2, default=str)[:1500])

print("\n=== strategy performance (S1/S2) ===")
st = get("/api/vantage/status")
for k in ("s1", "s2"):
    s = (st.get("report", {}).get("strategies", {}).get(k, {}) or {}).get("perf", {})
    print(" %s: closed=%s realized=$%s win=%s hit=%s expected=$%s cost=$%s client_equiv=$%s edge_vs_client=$%s" % (
        k, s.get("closed"), s.get("realized_usd"), s.get("win_rate"), s.get("hit_rate"),
        s.get("expected_usd"), s.get("cost_usd"), s.get("client_equiv_usd"), s.get("edge_vs_client_usd")))
    for r in (s.get("recent") or [])[:8]:
        print("    %s %-8s %-6s net=%.2f cost=%.2f client_equiv=%.2f exp=%.2f score=%s" % (
            r.get("close_time"), r.get("symbol"), r.get("stance"), r.get("net"), r.get("cost"),
            r.get("client_equiv"), r.get("expected"), r.get("score")))
