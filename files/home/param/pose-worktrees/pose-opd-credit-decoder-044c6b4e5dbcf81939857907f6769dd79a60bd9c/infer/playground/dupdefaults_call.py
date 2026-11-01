# 10-26 dupdefaults: ONE synthetic non-preempting live-mic call on instance P (client 'test', ?model=playground) with the page's FULL
# Live-mic default start params (incl. the duplex pose-defaults overlay), read from a headless page (= the livedup block's startMsg()).
# usage: python dupdefaults_call.py page <genweb URL>   (pagecheck venv)  |  python dupdefaults_call.py call <genweb URL> [seconds]   (qwen-s2 venv)
import asyncio, json, os, sys, time
G = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb"; LIVE = "/data/home/amartyavarma/ppm/aloe-posedup-live"
OUT = f"{G}/livedup_test/dupdefaults"; os.makedirs(OUT, exist_ok=True)
MODE = sys.argv[1]; U = sys.argv[2].strip().rstrip("/") + "/"; SECS = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
JS = """() => { const PAGE_ROWS=['preset','clip','person','persona','aec','ns','agc'], V=VALS, v={};
  for(const r of SCHEMA){ if(PAGE_ROWS.includes(r.name)) continue; let x=V[r.name]; if(x===undefined) x=r.default;
    if(r.b01) x=x?1:0; else if(r.name==='prefill_src') x=+x; else if(r.name==='seed') x=+x?+x:null; v[r.name]=x; }
  const P=window.livedup.LD.pre||{}, preset=String(V.preset), dc=(P.default_clip||{})[preset];
  const clip=V.clip==='same'||!V.clip?(dc||'none'):V.clip;
  return {model:MODEL, preset, clip, person:V.person, v}; }"""
async def page_defaults():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await (await b.new_context(viewport={"width": 1440, "height": 900})).new_page()
        await pg.goto(U + "?model=duplex"); await pg.wait_for_timeout(9000)
        d = await pg.evaluate(JS); await b.close(); return d
async def call(d):
    import aiohttp, numpy as np, soundfile as sf
    x, sr = sf.read(f"{G}/livedup_test/a.wav", dtype="float32", always_2d=True); x = x.mean(1)[:int(SECS * 24000)]; assert sr == 24000
    base = open(f"{LIVE}/URL.txt").read().split()[0].rstrip("/")
    v = d["v"]
    start = dict({"type": "start", "client": "test", "model": "playground", "preset": d["preset"], "persona": "", "starting_context": "",
                  "persona_only": True, "clip": d["clip"]}, **v, params=dict(v))
    if d.get("person") and d["person"] != "auto":
        start["person"] = d["person"]
    json.dump(start, open(f"{OUT}/start.json", "w"), indent=1)
    frames, msgs, ready, done = [], [], None, None
    async with aiohttp.ClientSession() as s:
        for _ in range(90):
            async with s.get(base + "/api/presets?model=playground") as r:
                pj = await r.json() if r.status == 200 else {}
            if r.status == 200 and not pj.get("busy"):
                break
            print("P busy / not ready: waiting 20 s", flush=True); await asyncio.sleep(20)
        else:
            raise SystemExit("P busy for 30 min")
        async with s.ws_connect(base.replace("https://", "wss://") + "/ws?model=playground", max_msg_size=8 << 20, heartbeat=20) as ws:
            await ws.send_str(json.dumps(start))
            while ready is None:
                m = await ws.receive(timeout=180)
                assert m.type == aiohttp.WSMsgType.TEXT, m
                dd = json.loads(m.data); msgs.append(dd)
                if dd["type"] == "route" and dd.get("model") != "playground":
                    await ws.send_str(json.dumps({"type": "stop"})); raise SystemExit(f"routed off P: {dd}")
                if dd["type"] == "ready": ready = dd
                elif dd["type"] == "error": raise SystemExit(f"error {dd}")
            async def sender():
                n, t1 = 1920, time.time()
                for i in range(0, len(x), n):
                    await ws.send_bytes(np.clip(x[i:i + n] * 32767, -32768, 32767).astype("<i2").tobytes())
                    dt = t1 + (i // n + 1) * 0.08 - time.time()
                    if dt > 0: await asyncio.sleep(dt)
                await ws.send_bytes(np.zeros(1920 * 3, "<i2").tobytes()); await asyncio.sleep(0.5)
                await ws.send_str(json.dumps({"type": "stop"}))
            snd = asyncio.ensure_future(sender())
            while True:
                m = await ws.receive(timeout=60)
                if m.type == aiohttp.WSMsgType.TEXT:
                    dd = json.loads(m.data)
                    if dd["type"] == "f": frames.append(dd)
                    else:
                        msgs.append(dd)
                        if dd["type"] in ("done", "error"): done = dd; break
                elif m.type != aiohttp.WSMsgType.BINARY: break
            snd.cancel()
    json.dump({"ready": ready, "done": done}, open(f"{OUT}/ready_done.json", "w"), indent=1, default=str)
    sid = ready.get("session"); time.sleep(3)
    sj = {}
    for f in ("session.json",):
        pth = f"{LIVE}/stage-lb/instP/sessions/{sid}/{f}"
        if os.path.exists(pth): sj = json.load(open(pth))
    ms = [f.get("ms") for f in frames if isinstance(f.get("ms"), (int, float))]
    K = ["pose_temp", "pose_topk", "pose_light_temp", "pose_light_topk", "pose_cfg_on", "pose_cfg_s", "pose_cfg_cb0", "pose_cfg_light",
         "pose_cfg_speaking_only", "pose_rep", "pose_rep_light", "pose_rep_win", "adaptive_rep_on", "pose_rep_cb0_speaking", "pose_rep_cb0_silent",
         "pose_rep_light_speaking", "pose_rep_light_silent"]
    ie = ready.get("in_effect") or {}; sp = sj.get("params") or {}
    lat = (done or {}).get("latency") or sj.get("latency") or {}
    rep = {"session": sid, "frames": len(frames), "in_effect_pose": {k: ie.get(k) for k in K}, "session_params_pose": {k: sp.get(k) for k in K},
           "latency_done": lat, "session_latency": sj.get("latency"),
           "client_ms_p50_p95": [float(np.percentile(ms, 50)), float(np.percentile(ms, 95))] if ms else None,
           "pcg_frames": [sum(1 for f in frames if f.get("pcg")), len(frames)],
           "session_keys": sorted(sj)[:60]}
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1, default=str); print(json.dumps(rep, default=str)[:4000])
if MODE == "page":
    d = asyncio.run(page_defaults()); json.dump(d, open(f"{OUT}/page_defaults.json", "w"), indent=1)
    print("page start params (pose):", {k: d["v"].get(k) for k in d["v"] if "pose" in k or "adaptive" in k}, d["preset"], d["clip"], d["model"])
else:
    asyncio.run(call(json.load(open(f"{OUT}/page_defaults.json"))))
