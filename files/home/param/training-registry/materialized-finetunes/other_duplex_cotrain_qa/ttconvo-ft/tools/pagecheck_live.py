import sys, time
from playwright.sync_api import sync_playwright
url = open("/data/home/amartyavarma/ppm/ttconvo-ft/PAGE_URL.txt").read().strip()
errs = []
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1400, "height": 1000})
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    r = pg.goto(url + "?t=" + str(int(time.time())), wait_until="networkidle", timeout=60000)
    pg.wait_for_timeout(4000)
    txt = pg.inner_text("body")
    pg.screenshot(path="/data/home/amartyavarma/ppm/ttconvo-ft/logs/page_shots/live-running-desktop.png", full_page=True)
    print("status", r.status, "errors", errs[:3])
    print("has job 26510:", "26510" in txt, "| has waiting:", "waiting for first metrics" in txt.lower())
    print(txt[:700].replace("\n", " | "))
    b.close()
