"""Queue ONE short run (clip_s 4) from the page and follow it: status pills, LIVE iframe, mp4 replacing it, words.
Usage: redesign_gen_check.py URL [local_html_or_-] [shots_dir]"""
import asyncio, json, os, sys, time
from urllib.parse import urlparse
from playwright.async_api import async_playwright

URL = sys.argv[1].strip().rstrip("/") + "/"
LOCAL = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
SHOTS = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")
HOST = urlparse(URL).netloc

async def main():
    html = open(LOCAL, encoding="utf-8").read() if LOCAL else None
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        ctx = await b.new_context(viewport={"width": 1440, "height": 900})
        pg = await ctx.new_page(); errs = []
        pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
        pg.on("console", lambda m: errs.append("console." + m.type + ": " + m.text) if m.type == "error" else None)
        if html is not None:
            async def handler(route):
                u = urlparse(route.request.url)
                if u.netloc == HOST and u.path in ("/", "/index.html"):
                    await route.fulfill(status=200, content_type="text/html; charset=utf-8", body=html)
                else:
                    await route.continue_()
            await pg.route("**/*", handler)
        await pg.goto(URL + "?v=gw2&clip_s=4")
        await pg.wait_for_function("SCHEMA && document.querySelectorAll('#form .row').length>10", timeout=60000); await pg.wait_for_timeout(800)
        print("clip_s", await pg.evaluate("VALS.clip_s"), "n_samples", await pg.evaluate("VALS.n_samples"))
        t0 = time.time(); await pg.click("#go"); seen = []; live_shot = False; last = None
        for i in range(240):
            await pg.wait_for_timeout(500)
            s = await pg.evaluate("""(()=>{ const f=document.querySelector('#dav iframe'), v=document.querySelector('#dav video'), ph=document.querySelector('#dav .ph');
              const box=(document.querySelector('#dav .frame')||{getBoundingClientRect:()=>({})}).getBoundingClientRect();
              return {sel:SEL, tab:(document.querySelector('.mtabs button.on')||{}).dataset?.m, pill:(document.getElementById('dpill')||{}).textContent||'', msg:(document.getElementById('dmsg')||{}).textContent||'',
                media:f?'iframe':v?('video '+v.src.split('/').pop()+' rs'+v.readyState):ph?('ph:'+ph.textContent):'none', frame:[Math.round(box.width),Math.round(box.height)],
                cap:(document.querySelector('#dav figcaption')||{}).textContent||'', ttff:window.__ttff, words:document.querySelectorAll('#words span[data-i]').length,
                hist:(document.querySelector('.run.sel .dot')||{}).className||''}; })()""")
            key = (s["pill"].split(" ")[0], s["media"].split(" ")[0])
            if key != last:
                last = key; seen.append(dict(s, t=round(time.time() - t0, 1)))
            if s["media"] == "iframe" and not live_shot and i > 3:
                live_shot = True; await pg.screenshot(path=f"{SHOTS}/1440x900-7-live.png")
            if s["media"].startswith("video") and "rs4" in s["media"] and s["pill"] == "Ready" and s["words"]:
                await pg.wait_for_timeout(1500); await pg.screenshot(path=f"{SHOTS}/1440x900-8-gen-ready.png"); seen.append(dict(s, t=round(time.time() - t0, 1))); break
        print(json.dumps(seen, indent=1)); print("errors", errs[:6]); await b.close()

asyncio.run(main())
