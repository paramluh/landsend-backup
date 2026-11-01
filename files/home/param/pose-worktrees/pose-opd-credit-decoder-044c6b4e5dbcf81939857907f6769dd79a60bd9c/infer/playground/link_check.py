import asyncio, sys
from playwright.async_api import async_playwright
URL = sys.argv[1].strip().rstrip("/") + "/"
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1400, "height": 1000}); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(URL + "?seed=1&n_samples=6&light_topk=127&pose_cfg_on=1"); await pg.wait_for_function("SCHEMA && document.querySelectorAll('#form .row').length>10", timeout=60000); await pg.wait_for_timeout(1200)
        q = await pg.evaluate("location.search")
        print("note:", await pg.evaluate("document.getElementById('linknote').textContent")[:90] if False else (await pg.evaluate("document.getElementById('linknote').textContent"))[:100])
        print("full link params:", len(q.split("&")), q[:160])
        print("kept:", await pg.evaluate("[VALS.seed, VALS.n_samples, VALS.light_topk, VALS.pose_cfg_on, VALS.prefill_src, VALS.prefill_s]"))
        await pg.evaluate("SEL=document.querySelector('.run') && RUNS[0].run_id; showRun(SEL)"); await pg.wait_for_timeout(2500)
        print("exact button:", await pg.evaluate("!!document.getElementById('exactlink')"), "errors", errs[:2]); await b.close()
asyncio.run(main())
