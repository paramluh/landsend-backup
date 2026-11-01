"""load a page file in headless Chromium: report page errors (syntax / runtime at load)."""
import asyncio, sys
from playwright.async_api import async_playwright


async def main(p):
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        pg = await b.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("file://" + p)
        await pg.wait_for_timeout(2500)
        ok = await pg.evaluate("typeof pcfgShow==='function' && typeof pcfgReset==='function' && typeof KIT==='object'")
        print("errors:", errs[:5]); print("functions ok:", ok)
        await b.close()

asyncio.run(main(sys.argv[1]))
