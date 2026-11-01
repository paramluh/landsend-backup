# 10-26: Show-GT toggle check (headless, desktop + phone): fresh load hides + never loads GT; toggle on shows it; links carry gt only by user
import asyncio, sys
from playwright.async_api import async_playwright
URL, RID = sys.argv[1].strip().rstrip("/") + "/", sys.argv[2]
async def one(b, name, vp):
    pg = await b.new_page(viewport=vp); errs, gtreq = [], []
    pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("request", lambda r: gtreq.append(r.url) if "/gt-" in r.url else None)
    await pg.goto(URL); await pg.wait_for_function("typeof SCHEMA!=='undefined' && SCHEMA", timeout=60000)
    await pg.evaluate(f"pickRun('{RID}')"); await pg.wait_for_timeout(2500)
    st = "(()=>{ const f=document.getElementById('dgt'), r=f.getBoundingClientRect(); return {visible:getComputedStyle(f).display!=='none'&&r.width>0, gt_video:!!f.querySelector('video'), ck:document.getElementById('gtck').checked, url_gt:new URLSearchParams(location.search).get('gt'), hscroll:document.documentElement.scrollWidth>innerWidth}; })()"
    print(name, "fresh:", await pg.evaluate(st), "gt requests", len(gtreq))
    await pg.click("#gtck"); await pg.wait_for_timeout(2500)
    print(name, "toggle on:", await pg.evaluate(st), "gt requests", len(gtreq), (await pg.evaluate("(document.querySelector('#dgt video')||{}).getAttribute?document.querySelector('#dgt video').getAttribute('src'):null")))
    await pg.goto(URL + "?v=gw2&model=monoplex&gt=1"); await pg.wait_for_function("typeof SCHEMA!=='undefined' && SCHEMA", timeout=60000); await pg.wait_for_timeout(500)
    print(name, "gt=1 without marker:", await pg.evaluate("[document.getElementById('gtck').checked, document.body.classList.contains('nogt')]"), "errors", errs[:2]); await pg.close()
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        await one(b, "desktop", {"width": 1366, "height": 820}); await one(b, "phone", {"width": 390, "height": 844}); await b.close()
asyncio.run(main())
