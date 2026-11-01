# 10-26: fresh page load (no params) -> Stream live is off -> a real single Generate -> prints the run id and the request body's stream fields
import asyncio, json, sys
from playwright.async_api import async_playwright
URL = sys.argv[1].strip().rstrip("/") + "/"
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1280, "height": 800}); errs, sent = [], []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("request", lambda r: sent.append(json.loads(r.post_data or "{}")) if r.url.endswith("/api/gen") else None)
        await pg.goto(URL); await pg.wait_for_function("typeof SCHEMA!=='undefined' && SCHEMA", timeout=60000); await pg.wait_for_timeout(800)
        print("fresh:", await pg.evaluate("[document.getElementById('streamck').checked, document.getElementById('streamhint').textContent, VALS.n_samples, location.search.includes('stream')]"))
        await pg.goto(URL + "?v=gw2&model=monoplex&n_samples=1&stream=1"); await pg.wait_for_function("typeof SCHEMA!=='undefined' && SCHEMA", timeout=60000); await pg.wait_for_timeout(800)
        print("old link stream=1 (no marker):", await pg.evaluate("[document.getElementById('streamck').checked, new URLSearchParams(location.search).get('stream')]"))
        await pg.goto(URL); await pg.wait_for_function("typeof SCHEMA!=='undefined' && SCHEMA", timeout=60000); await pg.wait_for_timeout(800)
        async with pg.expect_response(lambda r: r.url.endswith("/api/gen")) as ri:
            await pg.click("#go")
        r = await (await ri.value).json(); print("run", r.get("run_id"), "sent stream", [(x.get("stream"), x.get("stream_set_by_user")) for x in sent])
        print("errors", errs[:2]); await b.close()
asyncio.run(main())
