# 10-26 dupui: headless check of the dupui page block (read-only; never presses Generate).
# usage: python dupui_page_check.py <URL> <duplex run id> [--inject genweb/dupui_patches/dupui_block.html]
#   --inject: test the block on the live page before it is installed (adds its <style> + <script> after load).
import asyncio
import json
import re
import sys

from playwright.async_api import async_playwright

U = sys.argv[1].strip().rstrip("/") + "/"
RID = sys.argv[2]
INJ = open(sys.argv[sys.argv.index("--inject") + 1]).read() if "--inject" in sys.argv else ""
ROWS = """[{"name":"dup_mode","label":"generate","group":"Conversation","type":"enum","default":"duo","choices":["duo","replay"],"choice_labels":{"duo":"both sides","replay":"replay"},"hint":"x"},
 {"name":"prefill_line","label":"opening: partner audio","group":"Prefill","type":"enum","default":"aligned","choices":["aligned","delayed"],"choice_labels":{"aligned":"aligned","delayed":"delayed"},"hint":"y"}]"""


async def inject(pg):
    if not INJ:
        return
    css = re.search(r"<style>(.*?)</style>", INJ, re.S).group(1)
    js = re.search(r"<script>(.*?)</script>", INJ, re.S).group(1)
    await pg.add_style_tag(content=css)
    await pg.add_script_tag(content=js)


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        for w in (1280, 1440, 1920, 390):
            pg = await b.new_page(viewport={"width": w, "height": 900})
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            await pg.goto(U + "?model=duplex&dup_mode=duo")
            await pg.wait_for_timeout(5000)
            await inject(pg)
            await pg.wait_for_timeout(500)
            r = await pg.evaluate("""(rows)=>{ const has=(SCH.duplex||[]).some(r=>r.name==='dup_mode');
                if(!has){ SCH.duplex=SCH.duplex.concat(rows); if(MODEL==='duplex'){ SCHEMA=SCH.duplex; build(); } }
                const conv=[...document.querySelectorAll('#form .tabp[data-t=conv] .row')].map(e=>e.dataset.k);
                const pre=[...document.querySelectorAll('#form .tabp[data-t=prefill] .row')].map(e=>e.dataset.k);
                const o={schema_had_rows:has, conv, pre, gomodel:document.getElementById('gomodel').textContent,
                  a_after_dis:(document.querySelector('#form [data-k=a_after] select')||{}).disabled, pl_dis:(document.querySelector('#form [data-k=prefill_line] select')||{}).disabled,
                  url_dup:new URLSearchParams(location.search).get('dup_mode')};
                const s=document.getElementById('k-dup_mode'); s.value='replay'; s.onchange();
                o.replay=[document.getElementById('gomodel').textContent,(document.querySelector('#form [data-k=a_after] select')||{}).disabled,(document.querySelector('#form [data-k=prefill_line] select')||{}).disabled];
                s.value='duo'; s.onchange(); const c=document.getElementById('k-conv'); c.value='call1086'; c.onchange();
                o.call=[VALS.dup_mode,(document.getElementById('dupnote')||{}).textContent||null]; c.value='c0'; c.onchange(); s.value='duo'; s.onchange();
                o.after=[VALS.dup_mode,!!document.getElementById('dupnote'),document.getElementById('gomodel').textContent];
                c.value='call1086'; c.onchange(); c.value='c1'; c.onchange(); o.restore=[VALS.conv,VALS.dup_mode]; c.value='c0'; c.onchange();
                o.params=Object.keys(engineParams()).filter(k=>k==='dup_mode'||k==='prefill_line');
                return o; }""", json.loads(ROWS))
            print(w, "form:", r)
            r = await pg.evaluate("""async(rid)=>{ showMain('out'); pickRun(rid); await new Promise(r=>setTimeout(r,4000));
                const s=document.getElementById('dupconv'); const v=document.querySelector('#dav video[controls]')||document.querySelector('#dav video');
                const o={sec:!!s, h3:s&&s.querySelector('h3')?s.querySelector('h3').textContent:null, audio:s&&s.querySelector('audio')?s.querySelector('audio').getAttribute('src'):null,
                  dupk:(document.querySelector('#dkv .dupk')||{}).textContent||null, video:!!v};
                const rv=runVals({request:(await (await fetch('api/run/'+rid)).json()).request}); o.rerun_mode=rv.v.dup_mode;
                if(v){ v.muted=false; try{ await v.play(); }catch(e){ o.playErr=String(e); } await new Promise(r=>setTimeout(r,1500)); o.vt=v.currentTime;
                  const SA=(window.dupuiSync||{}).SA; o.sync=SA?{playing:!SA.paused, src:SA.getAttribute('src'), drift:+(SA.currentTime-v.currentTime).toFixed(3)}:null; }
                document.querySelector('#dcseg button[data-f="a.wav"]').click(); await new Promise(r=>setTimeout(r,500)); o.segSrc=document.getElementById('dcau').getAttribute('src');
                if(v) v.pause(); return o; }""", RID)
            print(w, "run view:", r)
            m = await pg.evaluate("""()=>({hscroll:document.documentElement.scrollWidth>innerWidth, sw:document.documentElement.scrollWidth, iw:innerWidth})""")
            print(w, "layout:", m, "errors:", errs[:3])
            await pg.close()
        pg = await b.new_page(viewport={"width": 1440, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(U)
        await pg.wait_for_timeout(5000)
        await inject(pg)
        await pg.wait_for_timeout(800)
        r = await pg.evaluate("""()=>({model:MODEL, gomodel:document.getElementById('gomodel').textContent, tabs:[...document.querySelectorAll('#form .tabs button')].filter(b=>!b.hidden).map(b=>b.textContent),
            hscroll:document.documentElement.scrollWidth>innerWidth, dupconv:!!document.getElementById('dupconv')})""")
        print("monoplex:", r, "errors:", errs[:3])
        await b.close()


asyncio.run(main())
