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
    ctx = b.new_context(viewport={"width": 1500, "height": 950})
    ctx.add_cookies([{"name": "zfx_session", "value": tok, "domain": "127.0.0.1", "path": "/"}])
    pg = ctx.new_page()
    errs = []
    pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type == "error" else None)
    pg.goto("http://127.0.0.1:8000/trading/antifraud", wait_until="domcontentloaded", timeout=45000)
    pg.wait_for_timeout(2500)
    # go to the Markout tab and trigger a load to catch the busy bar
    try:
        pg.click('.af-tab[data-af="markout"]'); pg.wait_for_timeout(800)
        pg.click('#mg-load'); pg.wait_for_timeout(700)   # catch the bar mid-load
        pg.screenshot(path=OUT + r"\ux_markout_loading.png")
        pg.wait_for_timeout(6000)                          # let it finish
        pg.screenshot(path=OUT + r"\ux_markout_done.png")
    except Exception as e:
        print("markout err", e)
    # check zfxRun exists
    has = pg.evaluate("() => typeof window.zfxRun === 'function' && typeof window.zfxLastGood === 'function'")
    print("zfxRun present:", has)
    print("console errors:", errs[:10] if errs else "none")
    b.close()
