import sys, urllib.request, json
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth
from playwright.sync_api import sync_playwright

with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])

def api(path):
    req = urllib.request.Request("http://127.0.0.1:8000" + path)
    req.add_header("Cookie", "zfx_session=" + tok)
    return json.loads(urllib.request.urlopen(req, timeout=40).read())

j = api("/api/vantage/status")
e = j["engine"]
print("ENGINE running=%s scored=%s acted=%s filled=%s blocked=%s" % (
    e["running"], e["scored"], e["acted"], e["filled"], e["blocked"]))
sigs = j.get("signals") or []
if sigs:
    print("stream newest:", sigs[0].get("traded"), sigs[0].get("symbol"), "score", sigs[0].get("score"))

with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1400, "height": 900})
    ctx.add_cookies([{"name": "zfx_session", "value": tok, "domain": "127.0.0.1", "path": "/"}])
    pg = ctx.new_page()
    errs = []
    pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type == "error" else None)
    pg.goto("http://127.0.0.1:8000/trading/antifraud", wait_until="domcontentloaded", timeout=45000)
    pg.wait_for_timeout(1500)
    pg.click('.af-tab[data-af="universe"]'); pg.wait_for_timeout(1200)
    print("antifraud console errors:", errs[:8] if errs else "none")
    b.close()
