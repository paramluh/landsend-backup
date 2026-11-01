"""aloe-posedup-live page check through the public tunnel (headless Chromium; = aloe-m1vox-live/stage-unbiased/unbiasedcheck.py adapted).
usage: pagecheck.py <URL> <24 kHz wav> <out dir> [secs]
1. plain URL, fresh browser: no page error, single model (no picker), every sampling control at the no-persona default + pose 1 / 30.
2. a short NON-PREEMPTING session from that page (client 'test'; refused while any session is active) with the Video avatar ON (a dummy
   player URL) and a person id: the start message carries the defaults + person; the server echoes them; the pose indicator counts frames
   with 32 codes; the avatar iframe src = <url>/?live=<session>; the page's own playback gain is 0 while the avatar is on, back on when off."""
import asyncio, json, sys
from playwright.async_api import async_playwright

URL = sys.argv[1].rstrip("/") + "/"
WAV, OUT = sys.argv[2], sys.argv[3]
SECS = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
AV = "https://avatar.invalid"
PERSON = "03_b437950e"
ARGS = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={WAV}",
        "--autoplay-policy=no-user-gesture-required"]
WSHOOK = """(()=>{ const _WS=window.WebSocket; window.__rx=[]; window.__tx=[];
  const W=function(u,p){ const w=p?new _WS(u,p):new _WS(u);
    w.addEventListener('message',e=>{ if(typeof e.data==='string'){ try{ const d=JSON.parse(e.data); if(d.type!=='f'&&d.type!=='pong') window.__rx.push(d); }catch(x){} } });
    const s=w.send.bind(w); w.send=x=>{ if(typeof x==='string'&&!x.includes('"ping"')) window.__tx.push(x); return s(x); }; return w; };
  W.prototype=_WS.prototype; W.CONNECTING=0; W.OPEN=1; W.CLOSING=2; W.CLOSED=3; window.WebSocket=W; })();"""
VALS = "(()=>Object.fromEntries(KIT.specs().filter(s=>s.key&&!s.soon).map(s=>[s.key,KIT.val(s)])))()"
WANT = {"factorised_text": True, "cat_temp": 1.0, "pad_bias": 0.0, "adaptive_pad_on": True, "pad_silent_s": 3.0, "pad_bias_silent": -0.1,
        "pad_bias_other": 0.0, "text_temp": 1.0, "text_topk": 0, "text_topp": 1.0, "minp": 0.05, "min_p": 0.05, "freq_pen": 0.3, "freq_win": 50,
        "text_rep": 1.0, "cb0_temp": 1.0, "cb0_topk": 99, "light_temp": 1.0, "light_topk": 79, "audio_rep": 1.0, "audio_rep_pen": 1.0,
        "adaptive_rep_on": False, "cfg_on": False, "cfg_cb0": False, "word_greedy": False, "pose_temp": 1.0, "pose_topk": 30,
        "pose_light_temp": 1.0, "pose_light_topk": 30}
R = {"checks": [], "errors": []}


def chk(name, ok, got=None):
    R["checks"].append({"ok": bool(ok), "check": name, **({"got": got} if got is not None else {})})


def same(a, b):
    if isinstance(b, bool):
        return (a in (True, 1, "1", "true")) == b
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


def diff(vals):
    return {k: vals[k] for k in vals if k in WANT and not same(vals[k], WANT[k])}


def flat(ie):
    f = {}
    for k, x in (ie or {}).items():
        f.update(x) if isinstance(x, dict) else f.__setitem__(k, x)
    return f


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=ARGS)
        ctx = await b.new_context(viewport={"width": 1280, "height": 1000}, permissions=["microphone"])
        await ctx.add_init_script(WSHOOK)
        pg = await ctx.new_page()
        pg.on("pageerror", lambda e: R["errors"].append(f"pageerror: {e}"))
        pg.on("console", lambda m: m.type == "error" and "avatar.invalid" not in m.text and R["errors"].append(f"console.error: {m.text}"))
        await pg.goto(URL, wait_until="domcontentloaded")
        await pg.wait_for_function("typeof PERSIST!=='undefined' && PERSIST.restored", timeout=60000)
        await pg.wait_for_timeout(1500)
        vals = await pg.evaluate(VALS)
        R["page_values"] = {k: vals[k] for k in vals if k in WANT}
        R["title"] = await pg.title()
        chk("every sampling control at the no-persona default (+ pose 1 / 30)", not diff(vals) and len(R["page_values"]) >= 20, diff(vals) or len(R["page_values"]))
        chk("single model: no picker", await pg.evaluate("$('modelrow').hidden"))
        chk("default preset = Luke (aloe-1-1648-0)", await pg.evaluate("$('preset').value") == "aloe-1-1648-0", await pg.evaluate("$('preset').value"))
        busy = await pg.evaluate("fetch('api/presets',{cache:'no-store'}).then(r=>r.json()).then(d=>d.busy)")
        if busy:
            R["session"] = "skipped: a session is active (test never preempts)"
        else:
            await pg.fill("#avurl", AV); await pg.dispatch_event("#avurl", "change")
            await pg.check("#avon"); await pg.fill("#person", PERSON); await pg.dispatch_event("#person", "change")
            await pg.evaluate("(()=>{ const o=window.startMsg; window.startMsg=()=>Object.assign(o(),{client:'test'}); })()")   # never preempts
            await pg.click("#go")
            try:
                await pg.wait_for_function("document.getElementById('go').textContent==='Stop'", timeout=90000)
                await pg.wait_for_timeout(int(SECS * 1000))
                tx = await pg.evaluate("window.__tx"); rx = await pg.evaluate("window.__rx")
                st0 = json.loads(tx[0]); rd = next((d for d in rx if d.get("type") == "ready"), {})
                ie = flat(rd.get("in_effect"))
                R["start_msg"] = {k: st0.get(k) for k in st0 if k in WANT or k in ("client", "person", "preset", "clip")}
                R["ready"] = {k: rd.get(k) for k in ("session", "person", "voice", "feed", "step", "label", "P")}
                chk("start msg: client test + defaults + person", st0.get("client") == "test" and not diff(st0) and st0.get("person") == PERSON, diff(st0))
                chk("server in_effect: the defaults", not diff(ie), diff(ie))
                chk("server ready: session id + person", bool(rd.get("session")) and rd.get("person") == PERSON, R["ready"])
                src = await pg.evaluate("$('avframe').src")
                chk("avatar iframe = <url>/?live=<session>, allow autoplay", src == f"{AV}/?live={rd.get('session')}" and
                    await pg.evaluate("$('avframe').getAttribute('allow')") == "autoplay" and not await pg.evaluate("$('avcard').hidden"), src)
                g_on = await pg.evaluate("OUT?OUT.g.gain.value:null")
                await pg.uncheck("#avon"); await pg.wait_for_timeout(600)
                g_off = await pg.evaluate("OUT?OUT.g.gain.value:null")
                chk("page playback muted while the avatar is on, restored when off", g_on is not None and g_on < 1e-3 and g_off > 0.5, [g_on, g_off])
                R["pose_indicator"] = await pg.evaluate("[$('posest').textContent, $('posetxt').textContent]")
                chk("pose indicator: frames with 32 codes", R["pose_indicator"][0] == "32 codes" and "frames with all 32 pose codes" in R["pose_indicator"][1], R["pose_indicator"])
                R["lane"] = await pg.evaluate("Array.from(document.querySelectorAll('#lane p.b')).map(p=>p.textContent).join(' | ').slice(0,400)")
                R["latency"] = await pg.evaluate("[$('latms').textContent, $('p95').textContent]")
                await pg.screenshot(path=f"{OUT}/session.png", full_page=True)
            finally:
                if (await pg.evaluate("document.getElementById('go').textContent")) == "Stop":
                    await pg.click("#go")
                await pg.wait_for_timeout(3000)
        await ctx.close(); await b.close()
    R["pass"] = all(c["ok"] for c in R["checks"]) and not R["errors"]
    json.dump(R, open(f"{OUT}/pagecheck.json", "w"), indent=1)
    print(json.dumps(R, indent=1))

asyncio.run(main())
