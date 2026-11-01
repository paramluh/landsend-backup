import asyncio, sys
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1400, "height": 1000}); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(sys.argv[1].rstrip("/") + "/?model=duplex"); await pg.wait_for_function("SCHEMA && document.querySelectorAll('#form .row').length>10", timeout=60000); await pg.wait_for_timeout(1500)
        print("model", await pg.evaluate("[MODEL, document.getElementById('mdl').value, !document.querySelector('#mdl option[value=duplex]').disabled]"), "rows", await pg.evaluate("document.querySelectorAll('#form .row').length"),
              "has conv", await pg.evaluate("!!document.querySelector('.row[data-k=conv]')"))
        await pg.evaluate("SEL='20261025-162423-f5bd'; showRun(SEL)"); await pg.wait_for_timeout(2500)
        print("video", await pg.evaluate("(()=>{ const v=document.querySelector('#dav video'); return v?[v.src.split('/').slice(-2).join('/'), v.readyState]:null; })()"), "errors", errs[:2]); await b.close()
asyncio.run(main())
