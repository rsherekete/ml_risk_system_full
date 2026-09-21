from playwright.sync_api import sync_playwright
import pathlib
html = pathlib.Path(r"c:\Users\RoyVivasi\Documents\notebook\proposal\taf_deck.html").as_uri()
out = r"c:\Users\RoyVivasi\Documents\notebook\proposal\.taf_tmp.pdf"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(html, wait_until="networkidle")
    pg.pdf(path=out, width="1280px", height="720px", print_background=True,
           margin={"top":"0","bottom":"0","left":"0","right":"0"})
    b.close()
import os
print("PDF:", out, os.path.getsize(out), "bytes")
