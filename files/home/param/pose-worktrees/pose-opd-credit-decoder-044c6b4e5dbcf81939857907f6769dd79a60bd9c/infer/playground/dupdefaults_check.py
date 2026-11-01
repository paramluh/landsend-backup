# 10-26 dupdefaults: headless check of the duplex pose-defaults overlay (read-only; never presses Start / Generate).
import asyncio, json, sys
from playwright.async_api import async_playwright
U = sys.argv[1].strip().rstrip("/") + "/"
K = ["pose_temp", "pose_topk", "pose_light_temp", "pose_light_topk", "pose_cfg_on", "pose_cfg_s", "pose_cfg_cb0", "pose_cfg_light", "pose_cfg_speaking_only",
     "pose_rep", "pose_rep_light", "pose_rep_win", "adaptive_rep_on", "pose_rep_cb0_speaking", "pose_rep_cb0_silent", "text_temp", "cb0_topk", "prefill_s"]
JS = "() => { const o={model:MODEL, n:(SCHEMA||[]).length}; for(const k of %s){ const r=(SCHEMA||[]).find(x=>x.name===k); o[k]=r?[VALS[k], r.default]:null; } o.url=location.search.length; return o; }" % json.dumps(K)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); ctx = await b.new_context(viewport={"width": 1440, "height": 900}); errs = []
        pg = await ctx.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)[:200]))
        for name, q in (("live fresh", "?model=duplex"), ("both-sides fresh", "?model=duplex&dup_mode=duo"), ("live link override", "?model=livedup&pose_topk=40&pose_cfg_on=0")):
            await pg.goto(U + q); await pg.wait_for_timeout(9000)
            print(name, json.dumps(await pg.evaluate(JS)))
        await pg.evaluate("() => document.getElementById('reset').click()"); await pg.wait_for_timeout(2500)
        print("after reset", json.dumps(await pg.evaluate(JS)))
        await pg.goto(U + "?model=duplex"); await pg.wait_for_timeout(9000)
        print("live settingsObj", json.dumps({k: v for k, v in (await pg.evaluate("() => settingsObj().params")).items() if k in K}))
        print("errors", errs); await b.close()
asyncio.run(main())
