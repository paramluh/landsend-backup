import asyncio, sys
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(sys.argv[1]); await pg.wait_for_timeout(3000); print("errors", errs[:3]); await b.close()
asyncio.run(main())
