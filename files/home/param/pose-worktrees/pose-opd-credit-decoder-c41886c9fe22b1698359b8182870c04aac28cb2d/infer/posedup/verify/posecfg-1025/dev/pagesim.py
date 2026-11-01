"""staged page + the staged server's schema (latency 95 ms set): Pose CFG group renders, default off, children gated; pcfgShow note."""
import asyncio, json, sys
from playwright.async_api import async_playwright


async def main(page, schema):
    rows = json.load(open(schema))
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        pg = await b.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("file://" + page)
        await pg.wait_for_timeout(1500)
        r = await pg.evaluate("""(rows)=>{ SCHEMA=rows; applySchema();
          const g=[...document.querySelectorAll('.grp h3')].map(h=>h.textContent);
          const C=document.querySelector('.ctl[data-k="pose_cfg_on"]'); const on=document.getElementById('p_pose_cfg_on');
          const kids=['pose_cfg_s','pose_cfg_cb0','pose_cfg_light'].map(k=>{ const e=document.getElementById('p_'+k); return [k, e?e.disabled:null, e?(e.type==='checkbox'?e.checked:e.value):null]; });
          const out={groups:g, has:!!C, checked:on?on.checked:null, kids, hint:C?C.querySelector('.hint').textContent:null};
          on.checked=true; on.dispatchEvent(new Event('change'));
          out.kids_after_tick=['pose_cfg_s','pose_cfg_cb0','pose_cfg_light'].map(k=>[k, document.getElementById('p_'+k).disabled]);
          for(let i=0;i<30;i++) pcfgShow({pcf:1, ms:70+i*0.1, lag:0.4});
          out.note_ok=document.querySelector('.ctl[data-k="pose_cfg_on"] .pcfglat').textContent;
          for(let i=0;i<300;i++) pcfgShow({pcf:1, ms:95, lag:0.4+i*0.05});
          const n=document.querySelector('.ctl[data-k="pose_cfg_on"] .pcfglat'); out.note_over=n.textContent; out.over=n.dataset.over;
          return out; }""", rows)
        print(json.dumps(r, indent=1)); print("errors:", errs[:5])
        await b.close()

asyncio.run(main(sys.argv[1], sys.argv[2]))
