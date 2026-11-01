import asyncio, sys
from playwright.async_api import async_playwright
URL = sys.argv[1].strip().rstrip("/") + "/"
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1400, "height": 1000}); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(URL + "?seed=1&n_samples=6&light_topk=127&pose_cfg_on=1"); await pg.wait_for_function("SCHEMA && document.querySelectorAll('.tabs button').length>5", timeout=60000); await pg.wait_for_timeout(1000)
        print("tabs", await pg.evaluate("[...document.querySelectorAll('.tabs button')].map(b=>b.textContent+(b.classList.contains('on')?'*':''))"))
        print("rows per tab", await pg.evaluate("[...document.querySelectorAll('.tabp')].map(p=>p.dataset.t+':'+p.querySelectorAll('.row').length)"))
        print("summary", (await pg.evaluate("document.getElementById('summary').textContent"))[:160])
        await pg.click(".tabs button[data-t=pose]"); await pg.wait_for_timeout(300)
        print("pose pair grid", await pg.evaluate("document.querySelectorAll('.tabp[data-t=pose] .pair .row').length"), "upload in actor:", await pg.evaluate("!!document.querySelector('.tabp[data-t=actor] #uplrow')"))
        print("url still full:", await pg.evaluate("location.search.split('&').length"), "errors", errs[:2]); await b.close()
asyncio.run(main())
