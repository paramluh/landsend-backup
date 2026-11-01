#!/usr/bin/env python3
"""posefix page check: serve the page dir on localhost, open it headless (playwright chromium), wait for the default clip's media, report
which audio/video elements loaded, and save a screenshot. usage: python shot.py <page dir> <out png> [port]"""
import asyncio
import functools
import http.server
import json
import sys
import threading

D, OUT = sys.argv[1], sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8765
srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), functools.partial(http.server.SimpleHTTPRequestHandler, directory=D))
threading.Thread(target=srv.serve_forever, daemon=True).start()


async def main():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        pg = await b.new_page(viewport={"width": 1280, "height": 1400})
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("requestfailed", lambda r: errs.append("failed " + r.url))
        pg.on("pageerror", lambda e: errs.append("pageerror " + str(e)))
        await pg.goto(f"http://127.0.0.1:{PORT}/index.html")
        await pg.wait_for_timeout(9000)
        st = await pg.evaluate("""() => ({clip: [...document.querySelectorAll('.rail button[aria-pressed=true]')].map(b => b.textContent),
            versions: [...document.querySelectorAll('#vers button')].map(b => b.textContent + (b.getAttribute('aria-pressed') === 'true' ? '*' : '')),
            video_hidden: document.getElementById('vids').hidden, video_ready: document.getElementById('vB').readyState,
            video_w: document.getElementById('vB').videoWidth, audio_ready: [document.getElementById('aA').readyState, document.getElementById('aB').readyState],
            params: document.getElementById('params').innerText.slice(0, 300), rail_groups: document.querySelectorAll('.grp').length})""")
        await pg.screenshot(path=OUT, full_page=False)
        print(json.dumps({"state": st, "errors": errs[:10]}, indent=1))
        await b.close()


asyncio.run(main())
srv.shutdown()
