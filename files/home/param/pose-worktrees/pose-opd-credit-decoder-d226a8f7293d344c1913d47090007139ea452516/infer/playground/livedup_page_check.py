# 10-26 livedup: headless check of the livedup page block (read-only: never presses Start).
# usage: python livedup_page_check.py <URL>
import asyncio, json, sys
from playwright.async_api import async_playwright
U = sys.argv[1].strip().rstrip("/") + "/"
STATE = """() => { const g=id=>document.getElementById(id), vis=e=>!!(e&&e.offsetParent!==null);
  const go=g('go'); return {model: typeof MODEL!=='undefined'?MODEL:null, url: location.search.slice(0,120), go: go&&go.textContent.trim(), goVis: vis(go),
    mode: vis(g('ldmode')), modeOn: [...document.querySelectorAll('#ldmode button.on')].map(b=>b.textContent), dlive: vis(g('dlive')), dav: vis(g('dav')), panel: vis(g('ldpanel')),
    tabs: [...document.querySelectorAll('#form .tabs button')].filter(b=>!b.hidden).map(b=>b.textContent), rows: document.querySelectorAll('#form [data-k]').length,
    preset: (typeof VALS!=='undefined'&&VALS)?VALS.preset:null, pfs: (typeof VALS!=='undefined'&&VALS)?VALS.prefill_s:null, plt: (typeof VALS!=='undefined'&&VALS)?VALS.pose_light_topk:null,
    src: window.livedup?window.livedup.LD.src:null, liveurl: window.livedup?window.livedup.LD.url:null,
    hscroll: document.documentElement.scrollWidth>window.innerWidth+1, sw: document.documentElement.scrollWidth, iw: window.innerWidth}; }"""
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for w in (1280, 1440, 1920, 390):
            ctx = await b.new_context(viewport={"width": w, "height": 900 if w > 400 else 800})
            pg = await ctx.new_page(); errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)[:200]))
            await pg.goto(U + "?model=duplex"); await pg.wait_for_timeout(6000)
            s1 = await pg.evaluate(STATE)
            print(w, "live:", json.dumps(s1))
            if w == 1280:
                await pg.evaluate("() => { const b=document.querySelector('#ldmode button[data-md=gen]'); if(b) b.click(); }"); await pg.wait_for_timeout(1500)
                print(w, "gen:", json.dumps(await pg.evaluate(STATE)))
                await pg.evaluate("() => { const b=document.querySelector('#ldmode button[data-md=live]'); if(b) b.click(); }"); await pg.wait_for_timeout(1500)
                print(w, "back:", json.dumps(await pg.evaluate(STATE)))
                rt = await pg.evaluate("""() => { const o=settingsObj('livedup'); const before=JSON.stringify(o.params);
                    VALS.pose_cfg_on=true; VALS.pose_cfg_speaking_only=true; urlSave(); const o2=settingsObj('livedup');
                    applySettings(o); const after=JSON.stringify(settingsObj('livedup').params); applySettings(o2); const after2=settingsObj('livedup').params;
                    return {n:Object.keys(o.params).length, roundtrip: before===after, changed_loaded: after2.pose_cfg_on===true&&after2.pose_cfg_speaking_only===true, url_has: location.search.includes('pose_cfg_speaking_only=1'), model: MODEL}; }""")
                print(w, "saveload:", json.dumps(rt))
                st = await pg.evaluate("() => { const m=window.livedup&&window.livedup.LD; return m?JSON.stringify(Object.assign({}, (function(){ const s=window.__ldsm; return {}; })())):null; }")
                sm = await pg.evaluate("() => { const r={}; try{ const f=window.livedup; } catch(e){} return null; }")
            await pg.goto(U); await pg.wait_for_timeout(5000)
            print(w, "mono:", json.dumps(await pg.evaluate(STATE)), "errors:", errs)
            await ctx.close()
        await b.close()
asyncio.run(main())
