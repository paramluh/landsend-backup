import asyncio, sys
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1400, "height": 1000}); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(sys.argv[1].rstrip("/") + "/?model=duplex"); await pg.wait_for_function("SCHEMA && document.querySelectorAll('.tabs button').length>5", timeout=60000); await pg.wait_for_timeout(800)
        print("dup tabs", await pg.evaluate("[...document.querySelectorAll('.tabp')].map(p=>p.dataset.t+':'+p.querySelectorAll('.row').length)"), "errors", errs[:2])
        await pg.goto(sys.argv[1].rstrip("/") + "/"); await pg.wait_for_function("SCHEMA && document.querySelectorAll('.tabs button').length>5", timeout=60000); await pg.wait_for_timeout(800)
        print("mono tabs", await pg.evaluate("[...document.querySelectorAll('.tabp')].map(p=>p.dataset.t+':'+p.querySelectorAll('.row').length)"), "errors", errs[:2]); await b.close()
asyncio.run(main())
