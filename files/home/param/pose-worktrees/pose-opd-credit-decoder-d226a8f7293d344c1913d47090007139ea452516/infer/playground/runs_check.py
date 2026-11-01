import asyncio, sys
from playwright.async_api import async_playwright
URL = sys.argv[1].strip().rstrip("/") + "/"; IDS = sys.argv[2].split(",")
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1400, "height": 1000}); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(URL + "?seed=1&n_samples=6"); await pg.wait_for_function("SCHEMA && document.querySelectorAll('.run').length>0", timeout=60000)
        for rid in IDS:
            await pg.evaluate(f"SEL='{rid}'; showRun('{rid}')"); await pg.wait_for_timeout(2500)
            v = await pg.evaluate("(()=>{ const v=document.querySelector('#dav video'); if(!v) return {video:false, ph:(document.querySelector('#dav')||{}).innerText}; return {video:true, src:v.src.split('/').pop(), ready:v.readyState, dur:+(v.duration||0).toFixed(1), msg:(document.getElementById('dh')||{}).textContent.slice(0,60)}; })()")
            print(rid, v)
        print("errors", errs[:3]); await b.close()
asyncio.run(main())
