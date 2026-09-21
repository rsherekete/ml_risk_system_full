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
    ctx = b.new_context(viewport={"width": 1400, "height": 950})
    ctx.add_cookies([{"name": "zfx_session", "value": tok, "domain": "127.0.0.1", "path": "/"}])
    pg = ctx.new_page()
    errs = []
    pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type == "error" else None)
    pg.goto("http://127.0.0.1:8000/replay", wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_timeout(4000)
    try:
        pg.click(".rp-play-btn")            # expand first card
        pg.wait_for_timeout(6000)           # let chart + events load
        pg.click(".rp-tp-play")             # hit play
        pg.wait_for_timeout(2500)           # let the playhead + account panel animate
        pg.screenshot(path=OUT + r"\replay_playing.png")
        # pause and capture frozen
        pg.click(".rp-tp-play")
        pg.wait_for_timeout(800)
        pg.screenshot(path=OUT + r"\replay_paused.png")
    except Exception as e:
        print("interaction err", e)
    print("console errors:", errs[:8] if errs else "none")
    b.close()
