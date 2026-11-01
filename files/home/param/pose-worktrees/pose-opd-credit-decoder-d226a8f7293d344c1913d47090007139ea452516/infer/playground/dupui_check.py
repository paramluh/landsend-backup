# 10-26 dupui audit: READ-ONLY headless check of the playground in DUPLEX mode (?model=duplex). Never presses Generate.
# usage: python dupui_check.py <URL> [<duplex run id>]
import asyncio, sys
from playwright.async_api import async_playwright
U = sys.argv[1].strip().rstrip("/") + "/"
RID = sys.argv[2] if len(sys.argv) > 2 else ""
STATE = """(()=>{ const q=new URLSearchParams(location.search), od=document.querySelector('#mdl option[value=duplex]');
  return {model:typeof MODEL!=='undefined'?MODEL:null, mdl:(document.getElementById('mdl')||{}).value, dupOpt:od?[od.disabled,od.textContent]:null,
    chip:(document.getElementById('mchip')||{}).textContent, gomodel:(document.getElementById('gomodel')||{}).textContent,
    tabs:[...document.querySelectorAll('#form .tabs button')].filter(b=>!b.hidden).map(b=>b.textContent),
    mtabs:[...document.querySelectorAll('.mtabs button')].map(b=>b.textContent.trim()),
    urlKeys:[...q.keys()].length, urlModel:q.get('model'), schemaN:(typeof SCHEMA!=='undefined'&&SCHEMA)?SCHEMA.length:0,
    missingInUrl:(typeof SCHEMA!=='undefined'&&SCHEMA)?SCHEMA.filter(r=>!q.has(r.name)).map(r=>r.name):null,
    gen:[...document.querySelectorAll('#pgset .fld, #pgset label')].map(e=>(e.querySelector('span')||e).textContent.trim()).slice(0,12),
    streamck:(document.getElementById('streamck')||{}).checked, gtck:(document.getElementById('gtck')||{}).checked, vw:(document.getElementById('vwsel')||{}).value,
    badge:(document.getElementById('vbadge')||{}).textContent, hscroll:document.documentElement.scrollWidth>innerWidth,
    sw:document.documentElement.scrollWidth, iw:innerWidth}; })()"""
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for w in (1280, 1440, 1920, 390):
            pg = await b.new_page(viewport={"width": w, "height": 900}); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
            await pg.goto(U + "?model=duplex"); await pg.wait_for_timeout(6000)
            print(w, "state:", await pg.evaluate(STATE))
            # model switch round-trip: do page flags (gt / stream / vw) survive a Model switch?
            r = await pg.evaluate("""(async()=>{ const o={}; if(typeof SCH==='undefined'||!SCH.monoplex||!SCH.duplex) return 'schemas not loaded';
                GT_ON=true; STREAM_X=true; VW_X='both'; urlSave(); o.before=location.search.match(/(gt|stream|vw)=[^&]*/g);
                const s=document.getElementById('mdl'); s.value='monoplex'; s.onchange(); await new Promise(r=>setTimeout(r,300));
                o.afterMono=[MODEL, location.search.match(/(gt|stream|vw)=[^&]*/g), GT_ON, STREAM_X, VW_X];
                s.value='duplex'; s.onchange(); await new Promise(r=>setTimeout(r,300)); o.afterDup=[MODEL, location.search.match(/(gt|stream|vw)=[^&]*/g)];
                return o; })()""")
            print(w, "switch:", r)
            if RID and w in (1280, 390):
                r = await pg.evaluate("""(async(rid)=>{ pickRun(rid); showMain('out'); await new Promise(r=>setTimeout(r,3500));
                    const d=document.getElementById('detail'); return {audios:[...d.querySelectorAll('audio')].map(a=>a.getAttribute('src')),
                      lanes:[...d.querySelectorAll('#words .lane > b')].map(b=>b.textContent), video:(document.querySelector('#dav video')||{}).src||null,
                      kv:(document.getElementById('dkv')||{}).textContent, gtShown:!!document.querySelector('#dgt video'), hdr:[...d.querySelectorAll('h3')].map(h=>h.textContent)}; })""", RID)
                print(w, "run view:", r)
                await pg.goto(U + "?model=duplex#batch=" + RID); await pg.wait_for_timeout(5000)
                r = await pg.evaluate("""(()=>({title:(document.getElementById('btitle')||{}).textContent, cards:document.querySelectorAll('#pbatch .bcard').length,
                    text:[...document.querySelectorAll('#pbatch .btx')].map(x=>x.textContent.slice(0,120)), zip:(document.getElementById('bzip')||{}).href||null,
                    hscroll:document.documentElement.scrollWidth>innerWidth}))()""")
                print(w, "batch:", r)
                await pg.evaluate("showMain('hist')"); await pg.wait_for_timeout(2500)
                r = await pg.evaluate("""(()=>({rows:document.querySelectorAll('#phist .hrow').length, dupRows:[...document.querySelectorAll('#phist .hrow .hp')].filter(x=>/Duplex/.test(x.textContent)).map(x=>x.textContent).slice(0,3),
                    filters:[...document.querySelectorAll('#phist select')].map(s=>s.getAttribute('aria-label'))}))()""")
                print(w, "history:", r)
            await pg.screenshot(path=f"/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/dupui-audit-{w}.png")
            print(w, "errors:", errs[:3])
            await pg.close()
        await b.close()
asyncio.run(main())
