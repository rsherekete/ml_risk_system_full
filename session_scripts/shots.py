"""Screenshot every key section for the design review."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\shots")
OUT.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:8000"
PAGES = [
    ("hub", "/hub", 4.0),
    ("executive", "/executive", 8.0),
    ("trading_summary", "/trading/summary", 12.0),
    ("trading_overview", "/trading/overview", 12.0),
    ("antifraud", "/trading/antifraud", 12.0),
    ("vantage", "/vantage", 8.0),
    ("quant_signals", "/quant/signals", 12.0),
    ("data", "/data", 8.0),
]

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 950})
    page.goto(BASE + "/login")
    page.fill("#username", "admin")
    page.fill("#password", "admin")
    page.click("#signin")
    page.wait_for_load_state("networkidle", timeout=30000)
    for name, path, settle in PAGES:
        try:
            page.goto(BASE + path, timeout=60000)
            try:
                page.wait_for_load_state("networkidle", timeout=int(settle * 1000))
            except Exception:
                pass
            page.wait_for_timeout(int(settle * 500))
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=False)
            print(f"{name}: OK")
        except Exception as e:
            print(f"{name}: ERR {type(e).__name__}: {e}")
    browser.close()
print("done ->", OUT)
