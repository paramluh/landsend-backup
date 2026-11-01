import asyncio, sys
from playwright.async_api import async_playwright
URL = sys.argv[1].strip().rstrip("/") + "/"
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1400, "height": 1000}); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(URL + "?pose_light_topk=99&prefill_s=3.2"); await pg.wait_for_function("SCHEMA && document.querySelectorAll('#form .row').length>10", timeout=60000)
        print("rows", await pg.evaluate("document.querySelectorAll('#form .row').length"), "groups", await pg.evaluate("[...document.querySelectorAll('details.g>summary')].map(s=>s.textContent)"))
        print("url vals", await pg.evaluate("[VALS.pose_light_topk, VALS.prefill_s, VALS.prefill_person, VALS.voice]"))
        await pg.wait_for_timeout(2500)
        print("runs listed", await pg.evaluate("document.querySelectorAll('.run').length"))
        await pg.evaluate("document.querySelector('.run') && document.querySelector('.run').click()"); await pg.wait_for_timeout(2500)
        print("detail", await pg.evaluate("document.getElementById('detail').innerText.slice(0,300)"))
        print("errors", errs[:3]); await b.close()
asyncio.run(main())
