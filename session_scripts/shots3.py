from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\shots")
BASE = "http://127.0.0.1:8000"
with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 950})
    page.goto(BASE + "/login")
    page.fill("#username", "admin"); page.fill("#password", "admin")
    page.click("#signin")
    page.wait_for_load_state("networkidle", timeout=30000)
    page.goto(BASE + "/trading/antifraud", timeout=60000)
    page.wait_for_timeout(6000)
    page.screenshot(path=str(OUT / "af_rail.png"))
    print("profiling OK")
    # rules pane
    page.click('.af-tab[data-af="rules"]')
    page.wait_for_timeout(5000)
    page.screenshot(path=str(OUT / "af_rules.png"))
    print("rules OK")
    # alerts pane
    page.click('.af-tab[data-af="alerts"]')
    page.wait_for_timeout(3000)
    page.screenshot(path=str(OUT / "af_alerts.png"))
    print("alerts OK")
    browser.close()
