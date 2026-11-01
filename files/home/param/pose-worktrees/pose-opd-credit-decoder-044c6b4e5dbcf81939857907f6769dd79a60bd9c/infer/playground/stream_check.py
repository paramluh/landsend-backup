# 10-26: Stream-live toggle check (headless). Intercepts api/gen (no real runs): auto default on for 1 sample, off for >1, explicit choice
# kept in the address (stream=0|1) and sent in the request body.
import asyncio, json, sys
from playwright.async_api import async_playwright
URL = sys.argv[1].strip().rstrip("/") + "/"
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1280, "height": 800}); errs, bodies = [], []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        async def gen(route):
            bodies.append(json.loads(route.request.post_data or "{}")); await route.fulfill(status=200, content_type="application/json", body=json.dumps({"run_id": "20000101-000000-%04x" % len(bodies)}))
        await pg.route("**/api/gen", gen)
        await pg.goto(URL + "?v=gw2&model=monoplex&n_samples=1"); await pg.wait_for_function("typeof SCHEMA!=='undefined' && SCHEMA && document.getElementById('streamck')", timeout=60000); await pg.wait_for_timeout(800)
        st = lambda: pg.evaluate("[document.getElementById('streamck').checked, document.getElementById('streamhint').textContent, new URLSearchParams(location.search).get('stream')]")
        print("n=1 auto:", await st())
        await pg.evaluate("(()=>{ const i=document.querySelector('[data-k=n_samples] input'); i.value=3; i.dispatchEvent(new Event('change')); })()"); await pg.wait_for_timeout(200)
        print("n=3 auto:", await st())
        await pg.click("#go"); await pg.wait_for_timeout(1500); print("n=3 bodies stream:", [x.get("stream") for x in bodies]); bodies.clear()
        await pg.click("#streamck"); await pg.wait_for_timeout(200); print("n=3 explicit on:", await st())
        await pg.click("#go"); await pg.wait_for_timeout(1500); print("n=3 explicit bodies stream:", [x.get("stream") for x in bodies]); bodies.clear()
        await pg.goto(URL + "?v=gw2&model=monoplex&n_samples=1&stream=0"); await pg.wait_for_function("typeof SCHEMA!=='undefined' && SCHEMA", timeout=60000); await pg.wait_for_timeout(800)
        print("link n=1 stream=0:", await st())
        await pg.click("#go"); await pg.wait_for_timeout(1500); print("link bodies stream:", [x.get("stream") for x in bodies])
        print("errors", errs[:3]); await b.close()
asyncio.run(main())
