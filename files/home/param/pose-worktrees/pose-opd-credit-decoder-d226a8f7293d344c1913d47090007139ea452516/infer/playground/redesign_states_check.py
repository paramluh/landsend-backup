"""Mock a run's transient states in the browser (no GPU): generating / rendering / failed; screenshot + pill geometry.
Usage: redesign_states_check.py URL [local_html_or_-] [shots_dir]"""
import asyncio, json, os, sys
from urllib.parse import urlparse
from playwright.async_api import async_playwright
URL = sys.argv[1].strip().rstrip("/") + "/"
LOCAL = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
SHOTS = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")
HOST = urlparse(URL).netloc; RID = "20261025-165621-3de6"; MODE = {"m": None}
def mock(d, m):
    d = dict(d); d["render"] = dict(d.get("render") or {})
    if m == "running":
        d.update(state="running", progress=0.37, headline=None, files=[], status={"state": "running", "msg": "generating 29/78 frames", "ms_per_frame": 71.2})
        d["render"].update(src=None, state=None, flash=False)
    elif m == "rendering":
        d.update(state="done"); d["render"].update(src=None, state="rendering", progress_s=3.1, seconds=8.04, flash=False)
    elif m == "error":
        d.update(state="error", headline=None, error="worker crashed: CUDA error: an illegal memory access was encountered (mock, long message to test wrapping inside the status line)", files=[])
        d["render"].update(src=None, state=None)
    return d
async def main():
    html = open(LOCAL, encoding="utf-8").read() if LOCAL else None
    async with async_playwright() as p:
        b = await p.chromium.launch(); out = {}
        for (w, h) in [(1440, 900), (390, 844)]:
            ctx = await b.new_context(viewport={"width": w, "height": h}); pg = await ctx.new_page(); errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            async def handler(route):
                u = urlparse(route.request.url)
                if html is not None and u.netloc == HOST and u.path in ("/", "/index.html"):
                    return await route.fulfill(status=200, content_type="text/html; charset=utf-8", body=html)
                if u.netloc == HOST and u.path == f"/api/run/{RID}" and MODE["m"]:
                    r = await route.fetch(); d = await r.json()
                    return await route.fulfill(status=200, content_type="application/json", body=json.dumps(mock(d, MODE["m"])))
                await route.continue_()
            await pg.route("**/*", handler)
            await pg.goto(URL); await pg.wait_for_function("SCHEMA && document.querySelectorAll('.run').length>0", timeout=60000)
            for m in ["running", "rendering", "error"]:
                MODE["m"] = m
                await pg.evaluate(f"SEL=null; document.getElementById('detail').dataset.id=''; SEL='{RID}'; showRun('{RID}'); showMain('out')"); await pg.wait_for_timeout(2500)
                g = await pg.evaluate("(()=>{const p=document.getElementById('dpill'), m=document.getElementById('dmsg'); const a=p.getBoundingClientRect(), b=m.getBoundingClientRect(); return {pill:p.textContent, pillW:Math.round(a.width), pillScroll:p.scrollWidth, pillRight:Math.round(a.right), msgLeft:Math.round(b.left), msgTop:Math.round(b.top), pillTop:Math.round(a.top), media:(document.querySelector('#dav iframe')?'iframe':document.querySelector('#dav video')?'video':(document.querySelector('#dav .ph')||{}).textContent), prog:!document.getElementById('dprog').hidden, sw:document.documentElement.scrollWidth}; })()")
                out[f"{w}-{m}"] = g
                await pg.screenshot(path=f"{SHOTS}/{w}x{h}-state-{m}.png")
            out[f"{w}-errors"] = errs[:4]; await ctx.close()
        await b.close(); print(json.dumps(out, indent=1))
asyncio.run(main())
