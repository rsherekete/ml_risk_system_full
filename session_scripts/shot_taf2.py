import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth
from playwright.sync_api import sync_playwright
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
OUT = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width":1500, "height":1050})
    ctx.add_cookies([{"name":"zfx_session","value":tok,"domain":"127.0.0.1","path":"/"}])
    pg = ctx.new_page()
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.goto("http://127.0.0.1:8000/taf", wait_until="domcontentloaded", timeout=45000)
    pg.wait_for_timeout(4500)
    pg.screenshot(path=OUT + r"\taf_overview.png", full_page=True)
    pg.click('.taf-t[data-t="monitor"]')
    pg.wait_for_timeout(5000)
    pg.screenshot(path=OUT + r"\taf_monitor.png", full_page=True)
    print("console errors:", errs[:8] if errs else "none")
    b.close()
