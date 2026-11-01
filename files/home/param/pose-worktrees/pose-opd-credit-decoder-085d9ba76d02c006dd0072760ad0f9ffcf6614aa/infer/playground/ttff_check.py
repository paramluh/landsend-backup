import asyncio, sys, time
from playwright.async_api import async_playwright
URL = sys.argv[1].strip().rstrip("/") + "/"
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required", "--site-per-process"])
        pg = await b.new_page(viewport={"width": 1400, "height": 1000}); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(URL + "?clip_s=10"); await pg.wait_for_function("SCHEMA && document.querySelectorAll('#form .row').length>10", timeout=60000)
        await pg.wait_for_timeout(1000); t0 = time.time()
        await pg.click("#go")
        for i in range(120):
            await pg.wait_for_timeout(500)
            v = await pg.evaluate("({ttff:window.__ttff, sel:SEL, media:(document.querySelector('#dav iframe')||document.querySelector('#dav video')||{}).src||'', head:(document.getElementById('dh')||{}).textContent||''})")
            if v["ttff"] is not None:
                break
        print("first avatar frame after Generate:", v["ttff"], "s; wall", round(time.time() - t0, 1), "s; run", v["sel"], "; media", v["media"][:90])
        print("headline", v["head"][:120]); print("errors", errs[:3]); await b.close()
asyncio.run(main())
