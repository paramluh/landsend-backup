import asyncio, functools, http.server, json, sys, threading
D, PAGE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8766), functools.partial(http.server.SimpleHTTPRequestHandler, directory=D))
threading.Thread(target=srv.serve_forever, daemon=True).start()
async def main():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        pg = await b.new_page(viewport={"width": 1280, "height": 1000})
        errs = []
        pg.on("pageerror", lambda e: errs.append("pageerror " + str(e)))
        await pg.goto(f"http://127.0.0.1:8766/{PAGE}")
        await pg.wait_for_timeout(4000)
        st = await pg.evaluate("""async () => { const vs=[...document.querySelectorAll('video')]; const v=vs[0]; v.muted=true; await v.play().catch(e=>0);
            await new Promise(r=>setTimeout(r,2500)); return {videos: vs.length, ready_first10: vs.slice(0,10).map(x=>x.readyState), playing_t: v.currentTime, w: v.videoWidth, src: v.getAttribute('src')} }""")
        await pg.screenshot(path=OUT)
        print(json.dumps({"state": st, "errors": errs[:5]}))
        await b.close()
asyncio.run(main()); srv.shutdown()
