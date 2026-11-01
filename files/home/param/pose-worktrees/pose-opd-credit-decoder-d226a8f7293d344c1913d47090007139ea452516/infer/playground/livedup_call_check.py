# 10-26 livedup: a REAL short call through the page with Chrome's fake mic (a wav as the microphone): Start -> ready -> frames / words /
# avatar iframe -> Stop. Client "playground" (never preempts); exits early if the backend reports busy.
# usage: python livedup_call_check.py <URL> <mic.wav> [seconds] [extra query]
import asyncio, json, sys
from playwright.async_api import async_playwright
U = sys.argv[1].strip().rstrip("/") + "/"; WAV = sys.argv[2]; SECS = float(sys.argv[3]) if len(sys.argv) > 3 else 15; EXTRA = sys.argv[4] if len(sys.argv) > 4 else ""
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={WAV}", "--autoplay-policy=no-user-gesture-required"])
        ctx = await b.new_context(viewport={"width": 1440, "height": 900}, permissions=["microphone"])
        pg = await ctx.new_page(); errs = []; posts = []
        pg.on("pageerror", lambda e: errs.append(str(e)[:200]))
        pg.on("request", lambda r: posts.append(r.url.split("/api/")[-1]) if "/api/livedup/" in r.url and r.method == "POST" else None)
        await pg.goto(U + "?model=livedup" + EXTRA); await pg.wait_for_timeout(6000)
        await pg.click("#go"); t = 0
        while t < 60:
            await pg.wait_for_timeout(1000); t += 1
            s = await pg.evaluate("() => { const L=window.livedup.LD; return {running:L.running, connecting:L.connecting, sid:L.sid, err:L.err, st:document.getElementById('ldst').textContent}; }")
            if s["running"] or s["err"] or (not s["connecting"] and not s["running"]): break
        print("after start:", json.dumps(s))
        if s["running"]:
            await pg.wait_for_timeout(int(SECS * 1000))
            r = await pg.evaluate("""() => { const L=window.livedup.LD, fr=document.getElementById('ldav'); return {frames:L.frames, ms_p50:L.ms.length?L.ms.slice().sort((a,b)=>a-b)[L.ms.length>>1]:null, lag:L.lag,
                words:document.getElementById('ldlane').innerText.slice(0,400), iframe:fr&&fr.src, ph:document.getElementById('ldph').hidden, st:document.getElementById('ldst').textContent,
                stats:document.getElementById('ldstats').textContent, in_effect:L.ready&&L.ready.in_effect?Object.fromEntries(Object.entries(L.ready.in_effect).filter(([k])=>/pose_cfg|prefill/.test(k))):null}; }""")
            print("live:", json.dumps(r))
            await pg.click("#go"); await pg.wait_for_timeout(5000)
            print("after stop:", json.dumps(await pg.evaluate("() => { const L=window.livedup.LD; return {running:L.running, go:document.getElementById('go').textContent.trim(), st:document.getElementById('ldst').textContent}; }")))
        print("posts:", posts, "errors:", errs)
        await b.close()
asyncio.run(main())
