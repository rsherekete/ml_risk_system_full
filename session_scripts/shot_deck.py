from playwright.sync_api import sync_playwright
import pathlib
html = pathlib.Path(r"c:\Users\RoyVivasi\Documents\notebook\proposal\taf_deck.html").as_uri()
OUT = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width":1280,"height":720})
    pg.goto(html, wait_until="networkidle")
    slides = pg.query_selector_all(".slide")
    print("slides:", len(slides))
    for i in (0,1):
        slides[i].scroll_into_view_if_needed()
        slides[i].screenshot(path=f"{OUT}\\deck_{i}.png")
    b.close()
print("done")
