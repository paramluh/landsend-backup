from playwright.sync_api import sync_playwright
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]); pg = b.new_page(viewport={"width": 1000, "height": 1100})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("https://queries-acer-thorough-note.trycloudflare.com/ppm_aloe.html"); pg.wait_for_timeout(2000)
    pg.screenshot(path="/data/home/amartyavarma/ppm/aloe-ft/tools/page_shot.png", full_page=True)
    print("charts", pg.evaluate("document.querySelectorAll('figure.chart').length"), "errors", errs)
    b.close()
