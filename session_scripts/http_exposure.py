import sys, urllib.request, json
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])

def get(path, timeout=90):
    req = urllib.request.Request("http://127.0.0.1:8000" + path)
    req.add_header("Cookie", "zfx_session=" + tok)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

j = get("/api/exposure/live")
print("LIVE exposure: rows=%d net=$%.0f gross=$%.0f errors=%s" % (
    len(j.get("rows", [])), j.get("total_net", 0), j.get("total_gross", 0), j.get("errors")))
for r in (j.get("rows") or [])[:6]:
    print("  %-10s net=$%.0f gross=$%.0f accts=%s" % (
        r.get("canonical_symbol"), r.get("net_notional") or 0,
        r.get("gross_notional") or 0, r.get("accounts")))
