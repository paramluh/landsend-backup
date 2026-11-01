# 10-26: headless check of Video weights "Both": the vw=both link, the main player's synced Param | Vansh pair (seek -> currentTime diff),
# the batch strip's split tiles and the batch grid's synced pairs, desktop + phone.
import asyncio, sys
from playwright.async_api import async_playwright
U, B, A = sys.argv[1].strip().rstrip("/") + "/", sys.argv[2], sys.argv[3]
PAIR = """(async(rid)=>{ pickRun(rid); for(let i=0;i<40&&!document.querySelector('#dav .vwpair video:nth-of-type(1)');i++) await new Promise(r=>setTimeout(r,250));
  for(let i=0;i<40&&document.querySelectorAll('#dav .vwpair video').length<2;i++) await new Promise(r=>setTimeout(r,250));
  const hs=[...document.querySelectorAll('#dav .vwpair .vwh')], vs=hs.map(h=>h.querySelector('video')), A=document.querySelector('#dav .vwpair .vwh[data-w="param"] video'), B=document.querySelector('#dav .vwpair .vwh[data-w="vansh"] video');
  if(!A||!B) return {labels:hs.map(h=>h.innerText), n:vs.filter(Boolean).length};
  for(let i=0;i<40&&(A.readyState<2||B.readyState<2);i++) await new Promise(r=>setTimeout(r,250));
  const out={labels:hs.map(h=>(h.querySelector('.vwl')||{}).textContent), ready:[A.readyState,B.readyState], dur:[A.duration,B.duration].map(x=>Math.round(x*100)/100), controls:[A.controls,B.controls], mutedB:B.muted};
  A.muted=true; await A.play().catch(e=>out.playErr=String(e)); await new Promise(r=>setTimeout(r,1500));
  out.playing=[!A.paused,!B.paused]; out.diffPlay=Math.round(Math.abs(A.currentTime-B.currentTime)*1000)/1000;
  A.currentTime=4.0; await new Promise(r=>setTimeout(r,700)); out.diffSeek=Math.round(Math.abs(A.currentTime-B.currentTime)*1000)/1000; out.tSeek=[A.currentTime,B.currentTime].map(x=>Math.round(x*100)/100);
  await new Promise(r=>setTimeout(r,1500)); out.diffAfter=Math.round(Math.abs(A.currentTime-B.currentTime)*1000)/1000;
  A.pause(); await new Promise(r=>setTimeout(r,400)); out.paused=[A.paused,B.paused]; A.currentTime=1.0; await new Promise(r=>setTimeout(r,600)); out.diffPausedSeek=Math.round(Math.abs(A.currentTime-B.currentTime)*1000)/1000;
  return out; })"""
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        for w in (1280, 390):
            pg = await b.new_page(viewport={"width": w, "height": 900}); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
            await pg.goto(U + "?vw=both"); await pg.wait_for_timeout(5000)
            print(w, "toggle:", await pg.evaluate("[document.getElementById('vwsel').value, document.getElementById('vwhint').textContent.slice(0,60), new URLSearchParams(location.search).get('vw')]"))
            print(w, "main pair:", await pg.evaluate(PAIR, A))
            print(w, "tiles:", await pg.evaluate("[...document.querySelectorAll('.btile')].map(t=>[t.querySelectorAll('.vwt img').length,[...t.querySelectorAll('.vwt img')].filter(i=>i.naturalWidth>0).length])"))
            await pg.screenshot(path=f"/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/stage-batch/both-main-{w}.png")
            await pg.goto(U + "#batch=" + B); await pg.wait_for_timeout(6000)
            r = await pg.evaluate("""(async()=>{ const cs=[...document.querySelectorAll('#pbatch .vwcmp')]; const o={pairs:cs.length, tags:[...document.querySelectorAll('#pbatch .vwtag')].map(x=>x.textContent)};
                const c=cs[0]; if(!c) return o; const vs=c.querySelectorAll('video'); o.n=vs.length; if(vs.length<2) return o; const A=vs[0], B=vs[1];
                for(let i=0;i<40&&(A.readyState<2||B.readyState<2);i++) await new Promise(r=>setTimeout(r,250));
                A.muted=true; await A.play().catch(()=>{}); await new Promise(r=>setTimeout(r,1200)); A.currentTime=3.0; await new Promise(r=>setTimeout(r,700));
                o.diffSeek=Math.round(Math.abs(A.currentTime-B.currentTime)*1000)/1000; o.playing=[!A.paused,!B.paused]; A.pause(); o.hscroll=document.documentElement.scrollWidth>innerWidth; return o; })()""")
            print(w, "batch grid:", r, "errors:", errs[:2])
            await pg.screenshot(path=f"/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/stage-batch/both-grid-{w}.png")
        await b.close()
asyncio.run(main())
