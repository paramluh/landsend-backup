# 10-26: open a run in the page's Output view on the public link, play its replay video muted-autoplay-safe and report readyState,
# duration and decoded audio bytes (proves the exact render plays with audio).
import asyncio, sys
from playwright.async_api import async_playwright
URL, RID = sys.argv[1].strip().rstrip("/") + "/", sys.argv[2]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"]); pg = await b.new_page(); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(URL); await pg.wait_for_function("typeof SCHEMA!=='undefined' && SCHEMA", timeout=60000)
        await pg.evaluate(f"pickRun('{RID}')"); await pg.wait_for_function("document.querySelector('#dav video')", timeout=20000)
        await pg.evaluate("(()=>{ const v=document.querySelector('#dav video'); v.muted=false; v.play(); })()"); await pg.wait_for_timeout(4000)
        print(await pg.evaluate("(()=>{ const v=document.querySelector('#dav video'); return {src:v.getAttribute('src'), ready:v.readyState, dur:v.duration, t:+v.currentTime.toFixed(2), paused:v.paused, audio_bytes:v.webkitAudioDecodedByteCount, live_iframe:!!document.querySelector('#dav iframe')}; })()"))
        print("errors", errs[:2]); await b.close()
asyncio.run(main())
