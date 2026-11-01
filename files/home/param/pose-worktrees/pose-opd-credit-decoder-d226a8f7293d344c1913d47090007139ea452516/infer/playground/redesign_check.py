"""Layout check for the playground page: screenshots at 1440x900, 1280x800 and a phone, overflow + truncation checks, video readiness,
URL length, exact link. Usage: redesign_check.py URL [local_html_or_-] [shots_dir]
local_html: serve this file instead of the live index.html (request interception); '-' = test the live page."""
import asyncio, json, os, sys
from urllib.parse import urlparse
from playwright.async_api import async_playwright

URL = sys.argv[1].strip().rstrip("/") + "/"
LOCAL = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
SHOTS = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")
MONO_RUN = "20261025-165621-3de6"     # finished monoplex run with GT + words
DUP_RUN = "20261025-170246-535f"      # finished duplex run (A + B words)
os.makedirs(SHOTS, exist_ok=True)
HOST = urlparse(URL).netloc

OVERFLOW_JS = """() => {
  const W = innerWidth, out = {scrollW: document.documentElement.scrollWidth, bodyScrollW: document.body.scrollWidth, innerW: W, scrollH: document.documentElement.scrollHeight, innerH: innerHeight};
  const bad = [];
  for (const e of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(e); if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const r = e.getBoundingClientRect(); if (!r.width && !r.height) continue;
    if (r.right > W + 0.5 || r.left < -0.5) bad.push((e.id ? '#' + e.id : e.tagName.toLowerCase() + '.' + [...e.classList].join('.')) + ' ' + Math.round(r.left) + '..' + Math.round(r.right));
  }
  const inner = [];
  for (const e of document.querySelectorAll('.mpanel,.tabbody,.left,.words,.right,main,#form,.tabp,.runs')) {
    if (e.offsetParent === null && getComputedStyle(e).position !== 'fixed') continue;
    if (e.scrollWidth > e.clientWidth + 1) inner.push((e.id || e.className) + ' ' + e.scrollWidth + '>' + e.clientWidth);
  }
  out.outside = bad.slice(0, 12); out.innerHoriz = inner; return out; }"""

SELECT_JS = """() => {
  const c = document.createElement('canvas').getContext('2d'), res = [];
  for (const s of document.querySelectorAll('select')) {
    if (!s.offsetParent) continue; const cs = getComputedStyle(s); c.font = cs.fontWeight + ' ' + cs.fontSize + ' ' + cs.fontFamily;
    let widest = 0, wtxt = ''; for (const o of s.options) { const w = c.measureText(o.textContent).width; if (w > widest) { widest = w; wtxt = o.textContent; } }
    const sel = s.options[s.selectedIndex] ? s.options[s.selectedIndex].textContent : '';
    const avail = s.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight) - 18;
    res.push({id: s.id, avail: Math.round(avail), widest: Math.round(widest), sel, truncSel: c.measureText(sel).width > avail, truncAny: widest > avail, wtxt});
  } return res; }"""


async def main():
    html = open(LOCAL, encoding="utf-8").read() if LOCAL else None
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        report = {}

        async def newpage(w, h, mobile=False):
            ctx = await b.new_context(viewport={"width": w, "height": h}, is_mobile=mobile, has_touch=mobile, device_scale_factor=1)
            try:
                await ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin=URL.rstrip("/"))
            except Exception as e:  # noqa: BLE001
                print("clipboard perm:", e)
            pg = await ctx.new_page(); errs = []
            pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
            pg.on("console", lambda m: errs.append("console." + m.type + ": " + m.text) if m.type == "error" else None)
            pg.on("dialog", lambda d: asyncio.ensure_future(d.dismiss()))
            if html is not None:
                async def handler(route):
                    u = urlparse(route.request.url)
                    if u.netloc == HOST and u.path in ("/", "/index.html"):
                        await route.fulfill(status=200, content_type="text/html; charset=utf-8", body=html)
                    else:
                        await route.continue_()
                await pg.route("**/*", handler)
            return ctx, pg, errs

        async def ready(pg, q=""):
            await pg.goto(URL + q)
            await pg.wait_for_function("SCHEMA && document.querySelectorAll('#form .row').length>10 && document.querySelectorAll('.run').length>0", timeout=60000)
            await pg.wait_for_timeout(800)

        for (w, h) in [(1440, 900), (1280, 800)]:
            tag = f"{w}x{h}"; R = report[tag] = {}
            ctx, pg, errs = await newpage(w, h)
            # 1. default view (fresh storage)
            await ready(pg)
            R["default_tab"] = await pg.evaluate("[...document.querySelectorAll('.mtabs button.on')].map(b=>b.dataset.m)")
            R["default_overflow"] = await pg.evaluate(OVERFLOW_JS)
            R["url_params"] = await pg.evaluate("location.search.split('&').length")
            R["default_text_left"] = await pg.evaluate("document.getElementById('left').innerText")
            R["queue_text"] = await pg.evaluate("document.getElementById('queue').innerText")
            R["linknote_hidden"] = await pg.evaluate("document.getElementById('linknote').hidden")
            await pg.screenshot(path=f"{SHOTS}/{tag}-1-default.png")
            R["sync_seed"] = await pg.evaluate("""(()=>{ const f=document.querySelector('#genknobs [data-k=seed] input'); f.value='7'; f.dispatchEvent(new Event('change'));
              const row=document.querySelector('.tabp .row[data-k=seed] input[type=number]'); const out={row:row&&row.value, url:new URLSearchParams(location.search).get('seed'), val:VALS.seed};
              row.value='0'; row.dispatchEvent(new Event('change')); out.back=f.value; out.url2=new URLSearchParams(location.search).get('seed'); return out; })()""")
            R["persona_open"] = await pg.evaluate("(document.querySelector('.row[data-k=persona] details')||{}).open")
            R["info_icons"] = await pg.evaluate("document.querySelectorAll('.info').length")
            # 2. Settings subtabs: selects not truncated, rows per tab
            sel = {}; rows = {}
            for t in await pg.evaluate("[...document.querySelectorAll('.tabs button')].filter(b=>!b.hidden).map(b=>b.dataset.t)"):
                await pg.click(f".tabs button[data-t={t}]"); await pg.wait_for_timeout(150)
                sel[t] = [s for s in await pg.evaluate(SELECT_JS) if s["id"] != "mdl" or t == "run"]
                rows[t] = await pg.evaluate(f"document.querySelectorAll('.tabp[data-t={t}] .row').length")
                ov = await pg.evaluate(OVERFLOW_JS)
                if ov["outside"] or ov["innerHoriz"] or ov["scrollW"] > ov["innerW"]:
                    R.setdefault("tab_overflow", {})[t] = ov
                if t == "actor":
                    await pg.screenshot(path=f"{SHOTS}/{tag}-3a-settings-actor.png")
            R["selects"] = {t: [x for x in v if x["truncAny"] or x["truncSel"]] for t, v in sel.items()}
            R["selects_all"] = {t: [(x["id"], x["avail"], x["widest"]) for x in v] for t, v in sel.items()}
            R["rows_per_tab"] = rows
            await pg.click(".tabs button[data-t=pose]"); await pg.wait_for_timeout(200)
            await pg.screenshot(path=f"{SHOTS}/{tag}-3-settings-pose.png")
            # 3. a finished run (Output)
            await pg.evaluate(f"document.querySelector('.run[data-id=\"{MONO_RUN}\"]').click()")
            await pg.wait_for_function("document.querySelector('#dav video') && document.querySelector('#dav video').readyState>=1", timeout=30000)
            await pg.click("#mt-out"); await pg.wait_for_timeout(2500)
            R["run_video"] = await pg.evaluate("(()=>{const v=document.querySelector('#dav video'), g=document.querySelector('#dgt video'); return {src:v.src.split('/').pop(), ready:v.readyState, w:v.videoWidth, h:v.videoHeight, paused:v.paused, t:+v.currentTime.toFixed(2), boxW:Math.round(v.getBoundingClientRect().width), boxH:Math.round(v.getBoundingClientRect().height), gt: g?{ready:g.readyState,w:g.videoWidth,h:g.videoHeight}:null}; })()")
            R["words_on"] = await pg.evaluate("document.querySelectorAll('#words span.on').length")
            R["run_overflow"] = await pg.evaluate(OVERFLOW_JS)
            R["status"] = await pg.evaluate("document.querySelector('.status').innerText")
            R["headline"] = await pg.evaluate("document.getElementById('dh').innerText")
            await pg.screenshot(path=f"{SHOTS}/{tag}-2-run.png")
            # exact link
            await pg.click("#exactlink"); await pg.wait_for_timeout(400)
            try:
                link = await pg.evaluate("navigator.clipboard.readText()")
                R["exact_link_params"] = len(link.split("?", 1)[1].split("&")) if "?" in link else 0
                R["exact_link_head"] = link[:200]
            except Exception as e:  # noqa: BLE001
                R["exact_link_err"] = str(e)[:200]
            R["exact_btn"] = await pg.evaluate("document.getElementById('exactlink').textContent")
            # refresh must not restart the video
            t0 = await pg.evaluate("document.querySelector('#dav video').currentTime"); await pg.wait_for_timeout(3200)
            R["video_kept_playing"] = await pg.evaluate(f"(()=>{{const v=document.querySelector('#dav video'); return {{t_before:{t0:.2f}, t_after:+v.currentTime.toFixed(2), paused:v.paused, ended:v.ended}}; }})()")
            # 4. History
            await pg.click("#mt-hist"); await pg.wait_for_timeout(3000)
            R["thumbs"] = await pg.evaluate("(()=>{const t=[...document.querySelectorAll('.thumb')]; const v=t.map(x=>x.querySelector('video')).filter(Boolean); return {rows:t.length, videos:v.length, ready:v.filter(x=>x.readyState>=1).length, muted:v.every(x=>x.muted), preload:[...new Set(v.map(x=>x.preload))], sel:(document.querySelector('.run.sel')||{}).dataset?.id}; })()")
            R["hist_overflow"] = await pg.evaluate(OVERFLOW_JS)
            await pg.screenshot(path=f"{SHOTS}/{tag}-4-history.png")
            R["errors"] = errs[:8]
            await ctx.close()

        # duplex at 1440: settings + a duplex run (two word lanes)
        R = report["duplex-1440"] = {}
        ctx, pg, errs = await newpage(1440, 900)
        await ready(pg, "?model=duplex")
        R["model"] = await pg.evaluate("[MODEL, document.getElementById('mdl').value]")
        R["rows_per_tab"] = await pg.evaluate("[...document.querySelectorAll('.tabp')].map(p=>p.dataset.t+':'+p.querySelectorAll('.row').length)")
        await pg.click(".tabs button[data-t=conv]"); await pg.wait_for_timeout(200)
        R["selects"] = [s for s in await pg.evaluate(SELECT_JS) if s["truncAny"] or s["truncSel"]]
        R["upload_shown"] = await pg.evaluate("!!document.querySelector('#form #uplrow')")
        await pg.screenshot(path=f"{SHOTS}/1440x900-5-duplex-settings.png")
        await pg.evaluate(f"document.querySelector('.run[data-id=\"{DUP_RUN}\"]').click()")
        await pg.wait_for_function("document.querySelector('#dav video') && document.querySelector('#dav video').readyState>=1", timeout=30000)
        await pg.click("#mt-out"); await pg.wait_for_timeout(2000)
        R["lanes"] = await pg.evaluate("document.querySelectorAll('#words .lane').length")
        R["overflow"] = await pg.evaluate(OVERFLOW_JS)
        await pg.screenshot(path=f"{SHOTS}/1440x900-6-duplex-run.png")
        R["errors"] = errs[:8]
        await ctx.close()

        # old short link note + phone
        R = report["phone-390"] = {}
        ctx, pg, errs = await newpage(390, 844, mobile=True)
        await ready(pg, "?light_topk=127&seed=3")
        R["linknote"] = await pg.evaluate("document.getElementById('linknote').hidden?'':document.getElementById('linknote').textContent")
        R["url_params"] = await pg.evaluate("location.search.split('&').length")
        R["vals"] = await pg.evaluate("[VALS.light_topk, VALS.seed]")
        R["overflow"] = await pg.evaluate(OVERFLOW_JS)
        await pg.screenshot(path=f"{SHOTS}/phone-1-default.png", full_page=True)
        await pg.evaluate(f"document.querySelector('.run[data-id=\"{MONO_RUN}\"]').click()"); await pg.click("#mt-out"); await pg.wait_for_timeout(3000)
        R["overflow_run"] = await pg.evaluate(OVERFLOW_JS)
        await pg.screenshot(path=f"{SHOTS}/phone-2-run.png", full_page=True)
        R["errors"] = errs[:8]
        await ctx.close()
        await b.close()
    print(json.dumps(report, indent=1))

asyncio.run(main())
