"""POSE CFG page check through the public URL (headless Chromium; = tools/pagecheck.py's session path, client 'test', never preempts):
1. page loads without errors; a 'Pose CFG' group with the toggle OFF (default), strength 3 / cb0 / cb1-31 on and greyed while off; the toggle's
   hint carries the server's measured per-frame cost vs the 80 ms budget;
2. a session from the page: after SW s the toggle is ticked on the page (live 'set'); the server echoes pose_cfg_on; frames carry pcf 1 from
   then on; the note under the toggle shows this call's ms/frame with pose CFG on.
usage: pcfg_pagecheck.py <URL> <24 kHz wav> <out dir> [secs] [switch s]"""
import asyncio, json, sys
from playwright.async_api import async_playwright

URL = sys.argv[1].rstrip("/") + "/"
WAV, OUT = sys.argv[2], sys.argv[3]
SECS = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
SW = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
ARGS = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={WAV}",
        "--autoplay-policy=no-user-gesture-required"]
WSHOOK = """(()=>{ const _WS=window.WebSocket; window.__rx=[]; window.__tx=[]; window.__pcf=[];
  const W=function(u,p){ const w=p?new _WS(u,p):new _WS(u);
    w.addEventListener('message',e=>{ if(typeof e.data==='string'){ try{ const d=JSON.parse(e.data); if(d.type==='f') window.__pcf.push([d.f,d.pcf,d.pcs,d.ms]); else if(d.type!=='pong') window.__rx.push(d); }catch(x){} } });
    const s=w.send.bind(w); w.send=x=>{ if(typeof x==='string'&&!x.includes('"ping"')) window.__tx.push(x); return s(x); }; return w; };
  W.prototype=_WS.prototype; W.CONNECTING=0; W.OPEN=1; W.CLOSING=2; W.CLOSED=3; window.WebSocket=W;
  const _Wk=window.Worker;                       // 10-25 the page's WebSocket lives in an upload Worker (WSW): record its traffic too
  const K=function(u,o){ const w=new _Wk(u,o);
    w.addEventListener('message',e=>{ const m=e.data; if(m&&m.k==='m'&&typeof m.d==='string'){ try{ const d=JSON.parse(m.d); if(d.type==='f') window.__pcf.push([d.f,d.pcf,d.pcs,d.ms]); else if(d.type!=='pong') window.__rx.push(d); }catch(x){} } });
    const pm=w.postMessage.bind(w); w.postMessage=(m,t)=>{ if(m&&m.k==='send'&&typeof m.d==='string'&&!m.d.includes('"ping"')) window.__tx.push(m.d); return t?pm(m,t):pm(m); }; return w; };
  K.prototype=_Wk.prototype; window.Worker=K; })();"""
CTL = """(()=>{ const g=k=>document.getElementById('p_'+k); const C=document.querySelector('.ctl[data-k="pose_cfg_on"]');
  return {groups:[...document.querySelectorAll('.grp h3')].map(h=>h.textContent), has:!!C, on:g('pose_cfg_on')&&g('pose_cfg_on').checked,
    kids:['pose_cfg_s','pose_cfg_cb0','pose_cfg_light'].map(k=>[k,g(k)?g(k).disabled:null,g(k)?(g(k).type==='checkbox'?g(k).checked:+g(k).value):null]),
    hint:C?C.querySelector('.hint').textContent:null, note:(C&&C.querySelector('.pcfglat'))?C.querySelector('.pcfglat').textContent:null,
    over:(C&&C.querySelector('.pcfglat'))?C.querySelector('.pcfglat').dataset.over:null}; })()"""
R = {"checks": [], "errors": []}


def chk(name, ok, got=None):
    R["checks"].append({"ok": bool(ok), "check": name, **({"got": got} if got is not None else {})})


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=ARGS)
        ctx = await b.new_context(viewport={"width": 1280, "height": 1000}, permissions=["microphone"])
        await ctx.add_init_script(WSHOOK)
        pg = await ctx.new_page()
        pg.on("pageerror", lambda e: R["errors"].append(f"pageerror: {e}"))
        pg.on("console", lambda m: m.type == "error" and R["errors"].append(f"console.error: {m.text}"))
        await pg.goto(URL, wait_until="domcontentloaded")
        await pg.wait_for_function("typeof PERSIST!=='undefined' && PERSIST.restored", timeout=60000)
        await pg.wait_for_timeout(1500)
        c0 = await pg.evaluate(CTL)
        R["before"] = c0
        chk("Pose CFG group + toggle present", c0["has"] and "Pose CFG" in c0["groups"], c0["groups"])
        chk("toggle OFF by default", c0["on"] is False, c0["on"])
        chk("strength 3, cb0 on, cb1-31 on, all greyed while off", c0["kids"] == [["pose_cfg_s", True, 3], ["pose_cfg_cb0", True, True], ["pose_cfg_light", True, True]], c0["kids"])
        chk("toggle hint carries the measured ms/frame vs the 80 ms budget", bool(c0["hint"]) and "ms/frame" in c0["hint"] and "80 ms" in c0["hint"], c0["hint"])
        busy = await pg.evaluate("fetch('api/presets',{cache:'no-store'}).then(r=>r.json()).then(d=>d.busy)")
        if busy:
            R["session"] = "skipped: a session is active (test never preempts)"
        else:
            await pg.evaluate("(()=>{ const o=window.startMsg; window.startMsg=()=>Object.assign(o(),{client:'test'}); })()")   # never preempts
            await pg.click("#go")
            try:
                await pg.wait_for_function("document.getElementById('go').textContent==='Stop'", timeout=90000)
                await pg.wait_for_timeout(int(SW * 1000))
                try:
                    await pg.check("#p_pose_cfg_on", timeout=3000)
                    R["tick"] = "click"
                except Exception:  # noqa: BLE001   (controls panel collapsed: the same change event from script)
                    await pg.evaluate("(()=>{ const e=document.getElementById('p_pose_cfg_on'); e.checked=true; e.dispatchEvent(new Event('change')); })()")
                    R["tick"] = "script change event"
                await pg.wait_for_timeout(int((SECS - SW) * 1000))
                tx = await pg.evaluate("window.__tx"); rx = await pg.evaluate("window.__rx"); fr = await pg.evaluate("window.__pcf")
                R["n_tx"], R["n_rx"], R["n_frames"] = len(tx), len(rx), len(fr)
                st0 = json.loads(next(x for x in tx if '"start"' in x))
                sets = [json.loads(x) for x in tx[1:] if '"set"' in x]
                echo = [d for d in rx if d.get("type") == "settings" and "pose_cfg_on" in (d.get("changed") or [])]
                rd = next((d for d in rx if d.get("type") == "ready"), {})
                on_f = [f for f, pcf, s, ms in fr if pcf]
                R.update(start_pose_cfg=st0.get("pose_cfg_on"), sets=sets, echo=[(d["f"], d["changed"], d.get("rejected")) for d in echo],
                         ready={k: rd.get(k) for k in ("session", "pose_cfg_rows", "pose_cfg_latency")}, frames=len(fr), pcf_frames=len(on_f),
                         first_pcf_frame=on_f[0] if on_f else None)
                chk("start message: pose CFG off", st0.get("pose_cfg_on") in (False, 0, None), st0.get("pose_cfg_on"))
                chk("page sent a live set with pose_cfg_on", any(s.get("pose_cfg_on") in (True, 1) for s in sets), sets)
                chk("server echoed pose_cfg_on (settings)", bool(echo), R["echo"])
                chk("frames with pcf 1 from the switch on, contiguous", bool(on_f) and on_f == list(range(on_f[0], fr[-1][0] + 1)), [len(on_f), on_f[:1]])
                c1 = await pg.evaluate(CTL)
                R["after"] = c1
                chk("note under the toggle: this call's ms/frame with pose CFG on", bool(c1["note"]) and "ms/frame" in c1["note"], c1["note"])
                chk("children enabled while on", all(k[1] is False for k in c1["kids"]), c1["kids"])
                await pg.screenshot(path=f"{OUT}/pcfg_session.png", full_page=True)
            finally:
                if (await pg.evaluate("document.getElementById('go').textContent")) == "Stop":
                    await pg.click("#go")
                await pg.wait_for_timeout(3000)
        await ctx.close(); await b.close()
    R["pass"] = all(c["ok"] for c in R["checks"]) and not R["errors"]
    json.dump(R, open(f"{OUT}/pcfg_pagecheck.json", "w"), indent=1)
    print(json.dumps(R, indent=1))

asyncio.run(main())
