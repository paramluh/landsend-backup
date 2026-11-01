import sys
from playwright.sync_api import sync_playwright
url, out = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox"])
    for w, sfx, cs in ((1200, "desk", "light"), (390, "phone", "dark")):
        pg = b.new_page(viewport={"width": w, "height": 900}, color_scheme=cs)
        pg.goto(url, wait_until="load", timeout=60000)
        pg.screenshot(path=f"{out}-{sfx}.png", full_page=True)
        sw = pg.evaluate("document.documentElement.scrollWidth")
        print(sfx, "scrollWidth", sw)
    b.close()
