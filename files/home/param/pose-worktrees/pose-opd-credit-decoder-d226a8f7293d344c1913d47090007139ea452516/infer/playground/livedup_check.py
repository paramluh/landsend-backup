#!/usr/bin/env python3
"""10-26 livedup CHECKS (genweb/LIVEDUP_PROMPT.md 'Verify'; dupui's checks fork). Read-only towards the live demo: talks ONLY to the live
proxy's model "playground" (instance P) with client 'test' (non-preempting: the server refuses it while a session is active; we wait).
  livedup_check.py prep                 genweb/livedup_test/a.wav = c0 stem_1 (24 kHz real speech) for the synthetic mic
  livedup_check.py wait                 poll <live>/api/presets?model=playground every 30 s until P answers (exit 0) or 40 min (exit 1)
  livedup_check.py call                 part 1: 25 s synthetic mic call with a changed knob set -> livedup_test/call/{ready,frames,result}.json
  livedup_check.py history <session>    part 2: genweb /api/livedup import of that session -> runs/<rid> files + render/out/<rid>.mp4
  livedup_check.py page                 part 3: playwright page checks (+ a 15 s fake-mic call through the page)
  livedup_check.py all                  wait + call + history + page
Every assertion -> livedup_test/results.json and the '## checks' section of genweb/LIVEDUP_STATUS.md (PASS / FAIL / INFO + evidence).
Run on a compute node (srun --jobid=30452 --overlap -n 1 --mem=4G --cpu-bind=none ...), pagecheck venv for 'page'."""
import asyncio
import glob
import json
import os
import sys
import time

G = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb"
LIVE = "/data/home/amartyavarma/ppm/aloe-posedup-live"
T = f"{G}/livedup_test"
RES = f"{T}/results.json"
STATUS = f"{G}/LIVEDUP_STATUS.md"
MODEL = "playground"
KNOBS = {"pose_cfg_on": 1, "pose_cfg_speaking_only": 1, "text_temp": 0.9}   # the changed knob set (pose CFG speaking-only + a text knob)


def url(f):
    return open(f).read().split()[0].rstrip("/")


def live_url():
    return url(f"{LIVE}/URL.txt")


def genweb_url():
    return url(f"{G}/URL.txt")


def record(part, name, ok, ev):
    """ok: True PASS, False FAIL, None INFO."""
    try:
        r = json.load(open(RES))
    except (OSError, ValueError):
        r = {}
    r[f"{part}: {name}"] = {"ok": ok, "ev": str(ev)[:400], "t": time.strftime("%H:%M:%S", time.gmtime())}
    os.makedirs(T, exist_ok=True)
    json.dump(r, open(RES + ".tmp", "w"), indent=1)
    os.replace(RES + ".tmp", RES)
    write_status(r)
    print(("PASS" if ok else "INFO" if ok is None else "FAIL"), part, name, "|", str(ev)[:300], flush=True)


def write_status(r):
    lines = ["## checks", f"(livedup_check.py, updated {time.strftime('%H:%M:%S', time.gmtime())} UTC; PASS / FAIL / INFO + evidence)", ""]
    for k, v in r.items():
        lines.append(f"- {'PASS' if v['ok'] else 'INFO' if v['ok'] is None else 'FAIL'} {k} ({v['t']}): {v['ev']}")
    sec = "\n".join(lines) + "\n"
    try:
        s = open(STATUS).read()
    except OSError:
        s = "# LIVEDUP status\n\n"
    i = s.find("## checks")
    if i >= 0:
        j = s.find("\n## ", i + 3)
        s = s[:i] + sec + (s[j + 1:] if j >= 0 else "")
    else:
        s = s.rstrip("\n") + "\n\n" + sec
    open(STATUS + ".tmp", "w").write(s)
    os.replace(STATUS + ".tmp", STATUS)


def prep():
    import numpy as np
    import soundfile as sf
    z = np.load("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c0.npz")
    x = z["stem_1"].astype(np.float32)[:int(40 * 24000)]
    os.makedirs(T, exist_ok=True)
    sf.write(f"{T}/a.wav", x, 24000, subtype="PCM_16")
    sf.write(f"{T}/a48.wav", np.repeat(x, 2), 48000, subtype="PCM_16")   # Chromium's fake mic file (48 kHz is safest)
    print("wrote", f"{T}/a.wav", round(len(x) / 24000, 1), "s")


async def presets(s):
    async with s.get(live_url() + f"/api/presets?model={MODEL}", timeout=20) as r:
        return r.status, (await r.json(content_type=None) if r.status == 200 else await r.text())


def is_playground(pj):
    """P answered: the proxy routed ?model=playground to a backend tagged playground (proxy adds 'model' / 'backend' fields)."""
    if not isinstance(pj, dict):
        return False
    blob = json.dumps({k: pj.get(k) for k in ("model", "backend", "lb", "instance", "name")}, default=str).lower()
    return "playground" in blob or "instance p" in blob


async def wait(max_s=2400):
    import aiohttp
    t0 = time.time()
    async with aiohttp.ClientSession() as s:
        while time.time() - t0 < max_s:
            try:
                st, pj = await presets(s)
                if st == 200 and is_playground(pj) and pj.get("state", "ready") == "ready":
                    record("P", "instance P answers ?model=playground", True,
                           {k: pj.get(k) for k in ("model", "backend", "state", "busy", "step", "default_preset")})
                    return 0
                print(time.strftime("%H:%M:%S"), "P not ready:", st, str({k: pj.get(k) for k in ("model", "backend", "state")} if isinstance(pj, dict) else pj)[:200], flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "presets error", repr(e)[:200], flush=True)
            await asyncio.sleep(30)
    record("P", "instance P answers ?model=playground", False, f"not up after {max_s} s")
    return 1


async def call(secs=25.0):
    """= LIVE/test_client.py run() (client 'test', 80 ms real-time chunks), + every message kept for the assertions."""
    import aiohttp
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(f"{T}/a.wav", dtype="float32", always_2d=True)
    x = x.mean(1)[:int(secs * 24000)]
    assert sr == 24000, sr
    out = f"{T}/call"
    os.makedirs(out, exist_ok=True)
    frames, bchunks, msgs = [], [], []
    base = live_url()
    async with aiohttp.ClientSession() as s:
        for _ in range(60):                                             # never preempt: wait while P is busy
            st, pj = await presets(s)
            if st == 200 and not pj.get("busy"):
                break
            print("P busy / not ready; waiting 20 s", flush=True)
            await asyncio.sleep(20)
        else:
            record("call", "P free for the test client", False, "busy for 20 min")
            return 1
        start = {"type": "start", "client": "test", "model": MODEL, "preset": pj["default_preset"], "persona": None,
                 "clip": (pj.get("default_clip") or {}).get(pj["default_preset"]), "params": dict(KNOBS)}
        if pj.get("person_default"):
            start["person"] = pj["person_default"]
        wsu = base.replace("https://", "wss://").replace("http://", "ws://") + f"/ws?model={MODEL}"
        async with s.ws_connect(wsu, max_msg_size=8 << 20, heartbeat=20) as ws:
            await ws.send_str(json.dumps(start))
            t0 = time.time()
            ready = None
            while ready is None:
                m = await ws.receive(timeout=180)
                if m.type == aiohttp.WSMsgType.TEXT:
                    d = json.loads(m.data)
                    msgs.append(d)
                    if d["type"] == "route" and d.get("model") != MODEL:   # never run on the live demo's own backends
                        await ws.send_str(json.dumps({"type": "stop"}))
                        record("call", "proxy routed the test call to instance P", False, f"route {d}: aborted")
                        return 1
                    if d["type"] == "route":
                        record("call", "proxy routed the test call to instance P", True, f"route {d}")
                    if d["type"] == "ready":
                        ready = d
                    elif d["type"] == "error":
                        record("call", "session starts", False, d)
                        return 1
                else:
                    record("call", "session starts", False, f"ws closed {m}")
                    return 1
            t_ready = time.time() - t0

            async def sender():
                n = 1920
                t1 = time.time()
                for i in range(0, len(x), n):
                    await ws.send_bytes(np.clip(x[i:i + n] * 32767, -32768, 32767).astype("<i2").tobytes())
                    dt = t1 + (i // n + 1) * 0.08 - time.time()
                    if dt > 0:
                        await asyncio.sleep(dt)
                await ws.send_bytes(np.zeros(1920 * 3, "<i2").tobytes())
                await asyncio.sleep(0.5)
                await ws.send_str(json.dumps({"type": "stop"}))
            snd = asyncio.ensure_future(sender())
            done = None
            while True:
                m = await ws.receive(timeout=60)
                if m.type == aiohttp.WSMsgType.BINARY:
                    bchunks.append(np.frombuffer(m.data, "<i2").astype(np.float32) / 32768)
                elif m.type == aiohttp.WSMsgType.TEXT:
                    d = json.loads(m.data)
                    if d["type"] == "f":
                        frames.append(d)
                    else:
                        msgs.append(d)
                        if d["type"] in ("done", "error"):
                            done = d
                            break
                else:
                    break
            snd.cancel()
    b = np.concatenate(bchunks) if bchunks else np.zeros(0, np.float32)
    sf.write(f"{out}/b.wav", b, 24000)
    json.dump(ready, open(f"{out}/ready.json", "w"), indent=1, default=str)
    with open(f"{out}/frames.jsonl", "w") as fo:
        for f in frames:
            fo.write(json.dumps(f) + "\n")
    json.dump(msgs, open(f"{out}/msgs.json", "w"), indent=1, default=str)
    sid = ready.get("session")
    open(f"{T}/last_session.txt", "w").write(str(sid))
    # ---- assertions
    record("call", "ready carries the session id", bool(sid), f"session {sid}, ready after {t_ready:.1f} s")
    avp = ready.get("avatar_path") or ready.get("avatar") or ready.get("avatar_url")
    record("call", "ready carries the avatar path", bool(avp), f"avatar_path {avp!r}; ready keys {sorted(ready)[:40]}")
    words = [f["tok"] for f in frames if f.get("tok") and not f.get("ctl")]
    seg = b[int(len(b) * 0.2):]
    rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    record("call", "B replies with speech (audio RMS + words)", len(b) > 24000 and rms > 0.003 and len(words) >= 3,
           f"B {len(b) / 24000:.1f} s, RMS {rms:.4f}, {len(frames)} frames, {len(words)} word pieces: {''.join(words)[:160]!r}; done={done and done.get('type')}")
    ie = ready.get("in_effect") or {}
    bad = {k: (v, ie.get(k)) for k, v in KNOBS.items() if ie.get(k) is None or abs(float(ie.get(k)) - float(v)) > 1e-6}
    record("call", "changed knobs in effect at ready (in_effect)", not bad, f"wanted {KNOBS}; mismatches {bad}; rejected {ready.get('rejected')}")
    spk = np.asarray([int(f.get("spk", 0)) for f in frames])
    pcf = np.asarray([int(f.get("pcf", 0)) for f in frames])
    record("call", "per-frame pcf (pose CFG switched on)", bool(pcf.all()), f"pcf=1 on {int(pcf.sum())}/{len(frames)} frames (pcf = pose_cfg_on; the gate is pcg)")
    if any("pcg" in f for f in frames):
        pcg = np.asarray([int(f.get("pcg", 0)) for f in frames])
        n11, n10, n01 = int(((pcg == 1) & (spk == 1)).sum()), int(((pcg == 1) & (spk == 0)).sum()), int(((pcg == 0) & (spk == 1)).sum())
        record("call", "per-frame pcg = spk (guided only while speaking, the server's own spk flag)", n10 == 0 and n01 == 0 and n11 > 0,
               f"frames {len(frames)}: speaking {int(spk.sum())}, silent {int((spk == 0).sum())}; pcg=1&spk=1 {n11}, pcg=1&spk=0 {n10}, pcg=0&spk=1 {n01}")
    else:
        record("call", "per-frame pcg = spk (guided only while speaking, the server's own spk flag)", False, "no 'pcg' field in the frame messages")
    dn = done or {}
    dl_ = dn.get("latency") if isinstance(dn.get("latency"), dict) else {}
    dg, dt = dn.get("cfg_guided_frames", dl_.get("cfg_guided_frames")), dn.get("cfg_frames_total", dl_.get("cfg_frames_total"))   # the server nests them in latency
    record("call", "done: cfg_guided_frames < cfg_frames_total", dg is not None and dt is not None and 0 < int(dg) < int(dt), f"done cfg_guided_frames {dg} / cfg_frames_total {dt}")
    # session recording (P's own sessions dir, else any LB sessions dir)
    sd = None
    for _ in range(30):
        c = glob.glob(f"{LIVE}/stage-lb/instP/sessions/{sid}/session.json") + glob.glob(f"{LIVE}/stage-lb/*/sessions/{sid}/session.json") + \
            glob.glob(f"{LIVE}/sessions/{sid}/session.json")
        if c:
            sd = os.path.dirname(c[0])
            break
        time.sleep(2)
    if not sd:
        record("call", "session recorded (session.json / npz / feed.jsonl)", False, f"no sessions/{sid}/session.json under instP / LB dirs")
        return 0
    sj = json.load(open(f"{sd}/session.json"))
    files = sorted(os.listdir(sd))
    record("call", "session recorded (session.json / npz / feed.jsonl)", {"session.json", "session.npz", "feed.jsonl"} <= set(files),
           f"{sd}: {files}")
    pr = sj.get("params") or {}
    bad = {k: (v, pr.get(k)) for k, v in KNOBS.items() if pr.get(k) is None or abs(float(pr.get(k)) - float(v)) > 1e-6}
    record("call", "recorded settings show the changed knobs (session.json params)", not bad, f"mismatches {bad}; params has {len(pr)} keys")
    sg, stt = sj.get("cfg_guided_frames"), sj.get("cfg_frames_total")
    record("call", "session.json: cfg_guided_frames < cfg_frames_total", sg is not None and stt is not None and 0 < int(sg) < int(stt), f"{sg} / {stt}")
    try:
        import numpy as np
        zz = np.load(f"{sd}/session.npz")
        if "pcg" in zz.files and "spk" in zz.files:
            pg_, sp_ = zz["pcg"].astype(int), (zz["spk"] > 0.5).astype(int)
            n = min(len(pg_), len(sp_))
            record("call", "session.npz pcg = spk", n > 0 and bool((pg_[:n] == sp_[:n]).all()), f"{n} frames, {int((pg_[:n] != sp_[:n]).sum())} mismatches, pcg=1 on {int(pg_.sum())}")
        else:
            record("call", "session.npz pcg = spk", False, f"npz keys {zz.files}")
    except Exception as e:  # noqa: BLE001
        record("call", "session.npz pcg = spk", False, repr(e)[:200])
    fs = sj.get("feed") or {}
    nfeed = sum(1 for _ in open(f"{sd}/feed.jsonl")) if os.path.exists(f"{sd}/feed.jsonl") else 0
    record("call", "feed reached videort (avatar stream)", bool(fs) and nfeed > 50,
           f"session.json feed stats {json.dumps(fs, default=str)[:200]}; feed.jsonl {nfeed} lines; FEED_URL {open(f'{LIVE}/stage-lb/instP/FEED_URL.txt').read().strip() if os.path.exists(f'{LIVE}/stage-lb/instP/FEED_URL.txt') else '?'}")
    if avp:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            u = live_url() + (avp if str(avp).startswith("/") else "/" + str(avp))
            try:
                async with s.get(u, timeout=20) as r:
                    txt = await r.text()
                    record("call", "avatar player URL serves", r.status == 200 and len(txt) > 200, f"GET {u} -> {r.status}, {len(txt)} B")
            except Exception as e:  # noqa: BLE001
                record("call", "avatar player URL serves", False, f"GET {u}: {e!r}"[:300])
    print("session dir", sd)
    return 0


async def history(sid, wait_mp4_s=420):
    """part 2: genweb's livedup import (livedup_api.py routes, read from the file) -> runs/<rid>."""
    import aiohttp
    api = f"{G}/livedup_api.py"
    if not os.path.exists(api):
        record("history", "genweb livedup_api.py exists", False, "genweb/livedup_api.py not written yet")
        return 1
    src = open(api).read()
    routes = sorted(set(__import__("re").findall(r'["\'](/api/livedup[^"\']*)["\']', src)))
    record("history", "livedup routes", None, routes)
    gw = genweb_url()
    rid = None
    async with aiohttp.ClientSession() as s:
        body = {"session": sid, "settings": dict(KNOBS), "params": dict(KNOBS), "model": "livedup", "client": "livedup_check"}
        for path in [r for r in routes if r.endswith("/start")] + [r for r in routes if r.endswith("/stop")]:
            try:
                async with s.post(gw + path, json=body, timeout=60) as r:
                    j = await r.json(content_type=None)
                    rid = rid or (j.get("run_id") or j.get("rid") if isinstance(j, dict) else None)
                    if rid:
                        body["run_id"] = rid
                    record("history", f"POST {path}", r.status == 200, f"{r.status} {json.dumps(j, default=str)[:200]}")
            except Exception as e:  # noqa: BLE001
                record("history", f"POST {path}", False, repr(e)[:200])
    if not rid:
        c = [d for d in glob.glob(f"{G}/runs/*/request.json") if sid in open(d).read()]
        rid = os.path.basename(os.path.dirname(sorted(c)[-1])) if c else None
    if not rid:
        record("history", "History entry runs/<rid> for the session", False, f"no run references session {sid}")
        return 1
    rd = f"{G}/runs/{rid}"
    need = ["a.wav", "b.wav", "mix.wav", "words.json", "summary.json"]
    for _ in range(60):
        if all(os.path.exists(f"{rd}/{n}") for n in need):
            break
        time.sleep(3)
    have = sorted(os.listdir(rd))
    record("history", "History entry has A/B/mix audio + words + summary", all(n in have for n in need), f"runs/{rid}: {have}")
    try:
        sm = json.load(open(f"{rd}/summary.json"))
        ps = {k: v for k, v in sm.items() if any(t in k.lower() for t in ("cb0", "gesture", "hand", "pose"))}
        if isinstance(sm.get("pose_stats"), dict):
            ps.update(sm["pose_stats"])
        for _ in range(40):
            if os.path.exists(f"{rd}/pose_stats.json"):
                ps.update(json.load(open(f"{rd}/pose_stats.json")))
                break
            time.sleep(3)
        okp = any("cb0" in k for k in ps) and any("gesture" in k for k in ps) and any("hand" in k for k in ps)
        record("history", "summary has pose stats (cb0 repeat, gestures/min, hands moving %)", okp, json.dumps(ps, default=str)[:300])
        stj = json.load(open(f"{rd}/status.json"))
        pct, pa = sm.get("pose_cfg_active_pct"), str(sm.get("pose_cfg_active"))
        record("history", "summary/headline show the applied pose-CFG percentage (not pending)", pct is not None and "pending" not in pa and "%" in (pa + str(stj.get("headline"))),
               f"pose_cfg_active_pct {pct}; pose_cfg_active {pa!r}; headline {str(stj.get('headline'))[:160]!r}")
        st = json.dumps(sm, default=str)
        record("history", "summary carries the call settings", all(k in st for k in KNOBS), f"keys {sorted(sm)[:30]}")
    except (OSError, ValueError) as e:
        record("history", "summary has pose stats (cb0 repeat, gestures/min, hands moving %)", False, repr(e))
    t0 = time.time()
    mp4 = f"{G}/render/out/{rid}.mp4"
    while time.time() - t0 < wait_mp4_s and not os.path.exists(mp4):
        time.sleep(5)
    record("history", "post-call avatar mp4 rendered (render/out/<rid>.mp4)", os.path.exists(mp4),
           f"{mp4} {'%.1f MB after %.0f s' % (os.path.getsize(mp4) / 1e6, time.time() - t0) if os.path.exists(mp4) else 'missing after %d s' % wait_mp4_s}")
    async with aiohttp.ClientSession() as s:
        try:
            async with s.get(gw + "/api/history?n=50", timeout=30) as r:
                h = await r.text()
            record("history", "History API lists the call", rid in h or sid in h, f"/api/history {'contains' if (rid in h or sid in h) else 'lacks'} {rid}")
        except Exception as e:  # noqa: BLE001
            record("history", "History API lists the call", False, repr(e)[:200])
    open(f"{T}/last_rid.txt", "w").write(rid)
    return 0


PAGE_STATE = """(()=>{ const q=new URLSearchParams(location.search), btns=[...document.querySelectorAll('button')].filter(b=>b.offsetParent!==null);
  const txt=b=>(b.textContent||'').trim();
  return {model:(typeof MODEL!=='undefined')?MODEL:null, url:location.search.length, urlKeys:[...q.keys()].length,
    start:btns.filter(b=>/^(start|stop)\\b/i.test(txt(b))).map(txt).slice(0,3), generate:btns.filter(b=>/^generate$/i.test(txt(b))).map(txt).slice(0,2),
    save:[...document.querySelectorAll('button,a,label')].filter(b=>/save (settings|file)|copy settings|load settings/i.test(txt(b))).map(txt).slice(0,5),
    knobs:['pose_cfg_on','pose_cfg_speaking_only','text_temp','prefill_s'].filter(k=>document.querySelector('[name="'+k+'"],[data-k="'+k+'"],[data-name="'+k+'"],#k_'+k+',#f_'+k)||(typeof SCHEMA!=='undefined'&&SCHEMA&&SCHEMA.some(r=>r.name===k))),
    urlHas:['pose_cfg_on','text_temp'].filter(k=>q.has(k)),
    hscroll:document.documentElement.scrollWidth>innerWidth, sw:document.documentElement.scrollWidth, iw:innerWidth}; })()"""


async def page(call_s=15):
    from playwright.async_api import async_playwright
    gw = genweb_url() + "/"
    wav = f"{T}/a48.wav"
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                          f"--use-file-for-fake-audio-capture={wav}", "--autoplay-policy=no-user-gesture-required"])
        for w in (1280, 1440, 1920, 390):
            for q in ("?model=duplex", "", "?model=duplex&dup_mode=duo"):
                ctx = await b.new_context(viewport={"width": w, "height": 900}, accept_downloads=True)
                pg = await ctx.new_page()
                errs = []
                pg.on("pageerror", lambda e, errs=errs: errs.append(str(e)))
                await pg.goto(gw + q)
                await pg.wait_for_timeout(6000)
                st = await pg.evaluate(PAGE_STATE)
                nm = "duplex live" if q == "?model=duplex" else "duplex offline (dup_mode=duo)" if q else "monoplex"
                record("page", f"{nm} {w}px: no sideways scroll, 0 page errors", (not st["hscroll"]) and not errs,
                       f"scrollWidth {st['sw']} / {st['iw']}; errors {errs[:2]}")
                if q == "?model=duplex" and w == 1280:
                    record("page", "duplex: Start/Stop button (not Generate)", bool(st["start"]) and not st["generate"], f"start {st['start']}, generate {st['generate']}")
                    record("page", "duplex: Generation tab has the live knobs", len(st["knobs"]) >= 3, f"found {st['knobs']}")
                    record("page", "duplex: URL carries the knobs", len(st["urlHas"]) == 2, f"url keys {st['urlKeys']}, has {st['urlHas']}")
                    record("page", "bare ?model=duplex opens Live mic", st["model"] == "livedup", f"MODEL {st['model']}")
                    record("page", "duplex: Copy / Save / Load settings controls", len(st["save"]) >= 2, f"{st['save']}")
                    await save_load(pg)
                    await page_call(pg, call_s)
                await ctx.close()
        await b.close()
    return 0


async def save_load(pg):
    """settingsio: Save settings (#sisave) -> file; change a knob (VALS + urlSave); Load settings (#siload -> #sifile file chooser) -> the values are back."""
    import re
    try:
        if not await pg.locator("#sisave").count():
            record("page", "Save -> Load settings round-trips", False, "no #sisave (settingsio block)")
            return
        before = json.loads(await pg.evaluate("JSON.stringify(VALS)"))
        async with pg.expect_download(timeout=15000) as dl:
            await pg.click("#sisave")
        d = await dl.value
        path = f"{T}/saved_settings.json"
        await d.save_as(path)
        saved = json.load(open(path))
        await pg.evaluate("(()=>{ VALS.text_temp=0.33; urlSave(); })()")
        mid = json.loads(await pg.evaluate("JSON.stringify(VALS)"))
        await pg.click("#siload")
        async with pg.expect_file_chooser(timeout=10000) as fc:
            await pg.click("#sifile")
        await (await fc.value).set_files(path)
        await pg.wait_for_timeout(2000)
        after = json.loads(await pg.evaluate("JSON.stringify(VALS)"))
        diff = {k: (before.get(k), after.get(k)) for k in set(before) | set(after) if before.get(k) != after.get(k)}
        note = await pg.evaluate("(document.getElementById('sionote')||{}).textContent||''")
        ok = not diff and mid.get("text_temp") == 0.33 and saved.get("model") == "livedup" and len(saved.get("params") or {}) >= 30
        record("page", "Save -> Load settings round-trips", ok,
               f"file {d.suggested_filename}: model {saved.get('model')}, {len(saved.get('params') or {})} params, ui {sorted(saved.get('ui') or {})}; "
               f"text_temp {before.get('text_temp')} -> 0.33 -> {after.get('text_temp')}; diffs after load {dict(list(diff.items())[:5])}; note {note!r}")
        uq = await pg.evaluate("location.search")
        record("page", "URL carries the loaded knobs", "text_temp=" in uq and "pose_cfg_on=" in uq and "model=livedup" in uq, f"{len(uq)} chars: {uq[:160]}")
    except Exception as e:  # noqa: BLE001
        record("page", "Save -> Load settings round-trips", False, repr(e)[:300])


async def page_call(pg, secs):
    """Start (livedup) with Chromium's fake mic (the c0 wav), ~secs s, Stop: frames + B audio arrive, the avatar iframe (#ldav) loads ?live=<sid>,
    and the call is imported into History (runs/<sid>)."""
    import re
    nb = {"bin": 0, "txt": 0, "urls": []}

    def on_ws(ws):
        nb["urls"].append(ws.url)
        ws.on("framereceived", lambda pl: nb.__setitem__("bin" if isinstance(pl, (bytes, bytearray)) else "txt", nb["bin" if isinstance(pl, (bytes, bytearray)) else "txt"] + 1))
    pg.on("websocket", on_ws)
    try:
        st = pg.locator("button:visible").filter(has_text=re.compile(r"^\s*start\b", re.I)).first
        if not await st.count():
            record("page", "fake-mic call through the page: B audio + avatar", False, "no visible Start button")
            return
        await st.click()
        await pg.wait_for_timeout(secs * 1000)
        ld = await pg.evaluate("""(()=>{ const L=(window.livedup||{}).LD||{}, f=document.getElementById('ldav');
            return {sid:L.sid||null, running:!!L.running, frames:L.frames||0, err:L.err||null, ldav:f?f.src:null, ldavHidden:f?f.hidden:null,
                    pill:(document.getElementById('ldpill')||{}).textContent||'', stats:((document.getElementById('ldstats')||{}).textContent||'').slice(0,160)}; })()""")
        frames_ok = [f.url for f in pg.frames if "avatar" in f.url and "live=" in f.url]
        sp = pg.locator("button:visible").filter(has_text=re.compile(r"^\s*stop\b", re.I)).first
        if await sp.count():
            await sp.click()
        await pg.wait_for_timeout(3000)
        fr = ld["frames"] if isinstance(ld["frames"], (int, float)) else len(ld["frames"] or [])
        ok = bool(ld["sid"]) and fr > 50 and bool(frames_ok) and not ld["err"]
        record("page", "fake-mic call through the page: B audio + avatar", ok,
               f"session {ld['sid']}, {fr} frames, ws seen {len(nb['urls'])} ({nb['bin']} binary / {nb['txt']} text frames visible to the page); avatar iframe {str(ld['ldav'])[:120]} "
               f"(loaded frames {frames_ok[:1]}); pill {ld['pill']!r}; err {ld['err']}")
        if ld["sid"]:
            open(f"{T}/page_session.txt", "w").write(ld["sid"])
            rd = f"{G}/runs/{ld['sid']}"
            for _ in range(60):
                if os.path.exists(f"{rd}/summary.json"):
                    break
                await asyncio.sleep(3)
            record("page", "page call imported into History (runs/<session>)", os.path.exists(f"{rd}/summary.json"),
                   f"{rd}: {sorted(os.listdir(rd)) if os.path.isdir(rd) else 'missing'}")
    except Exception as e:  # noqa: BLE001
        record("page", "fake-mic call through the page: B audio + avatar", False, repr(e)[:300])


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "prep":
        return prep()
    if cmd == "wait":
        return asyncio.run(wait())
    if cmd == "call":
        return asyncio.run(call())
    if cmd == "history":
        return asyncio.run(history(sys.argv[2] if len(sys.argv) > 2 else open(f"{T}/last_session.txt").read().strip()))
    if cmd == "page":
        return asyncio.run(page())
    if cmd == "all":
        if asyncio.run(wait()):
            return 1
        asyncio.run(call())
        asyncio.run(history(open(f"{T}/last_session.txt").read().strip()))
        return asyncio.run(page())
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
