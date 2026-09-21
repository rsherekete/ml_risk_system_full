import sys, time
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

    # Hub
    pg.goto("http://127.0.0.1:8000/hub", wait_until="networkidle", timeout=45000)
    pg.wait_for_timeout(1500)
    pg.screenshot(path=OUT + r"\hub.png")
    # open client zoom
    try:
        pg.click("#zoom-open"); pg.wait_for_timeout(2500)
        pg.screenshot(path=OUT + r"\hub_zoom.png")
    except Exception as e:
        print("zoom err", e)

    # Replay
    pg.goto("http://127.0.0.1:8000/replay", wait_until="networkidle", timeout=60000)
    pg.wait_for_timeout(4000)
    pg.screenshot(path=OUT + r"\replay.png", full_page=True)
    # expand first card
    try:
        pg.click(".rp-play-btn")
        pg.wait_for_timeout(5000)
        pg.screenshot(path=OUT + r"\replay_open.png", full_page=True)
    except Exception as e:
        print("expand err", e)

    print("CONSOLE ERRORS:", errs[:20] if errs else "none")
    b.close()
print("done")
