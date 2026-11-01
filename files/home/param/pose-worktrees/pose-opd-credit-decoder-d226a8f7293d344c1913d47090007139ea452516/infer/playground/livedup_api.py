"""livedup (10-26 dupui): the playground's LIVE duplex calls. Start / Stop streams the mic to instance P of the live demo's own server.py (model
"playground" behind the live front proxy; the demo's preset is never touched: each call's knobs ride in its start message). Every call is
recorded into a run dir (run id = P's session id) so History, replay, pose stats (posefix's scorer job picks up tokens.npz + summary.json) and
the post-call renders work as for generated runs. Plumbing only: nothing here samples or decodes.
Mounted by server.py: import livedup_api; livedup_api.setup(app, globals())
  GET  /api/livedup/config  -> {live_url, alt_urls, presets, schema}: the live front's URL (URL.txt line 1; ALT_URLS.txt) and P's
                               /api/presets?model=playground + /api/params_schema?model=playground fetched HERE (the live proxy sends no CORS
                               headers), cached 10 s; each falls back to the URL without ?model; a total failure -> {"error": ...} in its place
  POST /api/livedup/start {session, settings, url_query, video_weights, vansh_steps, gt, label, person, prefill_s}
                            -> {run_id}: runs/<session id>/request.json (model "livedup", ui = settings: History re-run / copy-settings),
                               status 'live'. Idempotent (a second call only updates the settings).
  POST /api/livedup/stop  {session} -> {run_id, state}: the page ended the call; the import starts as soon as P has written the session's files
  GET  /api/livedup/run/{session} -> {run_id, state, msg, headline}
IMPORTER (thread, every 3 s): each P session (stage-lb/instP/sessions/<id>) with session.json (written at the call's end, also when the page
died: P's 45 s dead-session expiry) whose run is not done -> livedup_import.py in a subprocess (one at a time, nice 10) -> a.wav (mic), b.wav,
mix.wav, words.json (B's own lane; no ASR), tokens.npz, feed.jsonl, summary.json, status done; then the render jobs exactly as the workers /
gen() queue them (Param: render/jobs; Vansh: vdir(vsteps); Both: Param + compare.json -> server.comparer queues the Vansh re-render).
Calls the page never registered (it died before /start) are imported too, with the settings from session.json. A call with no session.json
45 min after its start is marked failed."""
import asyncio
import json
import os
import re
import subprocess
import threading
import time

import aiohttp
from aiohttp import web

C = {}
LIVE = "/data/home/amartyavarma/ppm/aloe-posedup-live"
P_SESS = f"{LIVE}/stage-lb/instP/sessions"
OTHER_SESS = (f"{LIVE}/stage-lb/instS4/sessions", f"{LIVE}/sessions", f"{LIVE}/stage-lb/instB/sessions", f"{LIVE}/stage-lb/instC/sessions")   # 10-26 main: instS4 = the slinky-4 demo instance (P parked)   # only for runs the page registered
PY = "/data/home/amartyavarma/venvs/qwen-s2/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
CFG_CACHE = {}
FAILED = {}                                                            # session -> (session.json mtime, tries)
LOCK = threading.Lock()


def _line(path, n=None):
    try:
        ls = [x.strip() for x in open(path).read().splitlines() if x.strip()]
    except OSError:
        return [] if n is None else None
    return ls if n is None else (ls[n] if len(ls) > n else None)


async def _fetch_json(sess, base, path):
    err = None
    for q in ("?model=playground", ""):
        try:
            async with sess.get(f"{base.rstrip('/')}{path}{q}", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    return await r.json(content_type=None)
                err = f"{path}{q}: HTTP {r.status}"
        except Exception as e:  # noqa: BLE001
            err = f"{path}{q}: {e!r}"[:300]
    return {"error": err}


async def config(req):
    hit = CFG_CACHE.get("c")
    if hit and time.time() - hit[0] < 10:
        return web.json_response(hit[1], headers={"Cache-Control": "no-store"})
    live = _line(f"{LIVE}/URL.txt", 0)
    out = {"live_url": live, "alt_urls": _line(f"{LIVE}/ALT_URLS.txt"), "model": "playground", "client": "playground"}
    if live:
        async with aiohttp.ClientSession() as s:
            out["presets"], out["schema"] = await asyncio.gather(_fetch_json(s, live, "/api/presets"), _fetch_json(s, live, "/api/params_schema"))
    else:
        out["presets"] = out["schema"] = {"error": "no live URL (URL.txt)"}
    CFG_CACHE["c"] = (time.time(), out)
    return web.json_response(out, headers={"Cache-Control": "no-store"})


def _session_dir(sid):
    for d in (P_SESS,) + OTHER_SESS:
        if os.path.isdir(f"{d}/{sid}"):
            return f"{d}/{sid}"
    return None


def _engine_params(settings, sj=None):
    """the engine knobs of a call: P's recorded effective params (session.json) when known, else the settings' numeric / bool values"""
    if sj and isinstance(sj.get("params"), dict):
        return dict(sj["params"])
    sc = (CFG_CACHE.get("c") or (0, {}))[1].get("schema")
    rows = sc if isinstance(sc, list) else ((sc or {}).get("rows") or (sc or {}).get("schema") or []) if isinstance(sc, dict) else []
    names = {r.get("name") for r in rows if isinstance(r, dict)}
    return {k: v for k, v in (settings or {}).items() if (k in names if names else isinstance(v, (int, float, bool)))}


def _register(sid, d, sj=None):
    """create / update runs/<sid>/request.json (+ status 'live' unless already further) from the page's start body d (or session.json sj)"""
    RUNS, jread, jwrite = C["RUNS"], C["jread"], C["jwrite"]
    rd = f"{RUNS}/{sid}"
    os.makedirs(rd, exist_ok=True)
    old = jread(f"{rd}/request.json", {}) or {}
    settings = d.get("settings") if isinstance(d.get("settings"), dict) else (old.get("ui") or {})
    if sj and not settings:
        cfg = sj.get("cfg") if isinstance(sj.get("cfg"), dict) else {}
        settings = dict(sj.get("params") or {}, **{k: cfg[k] for k in ("preset", "person", "prefill_s", "prefill_src", "pose_prefill", "prefill_audio",
                                                                          "prefill_words", "seed") if k in cfg})
    vw = str(d.get("video_weights") or old.get("video_weights") or settings.get("video_weights") or "param").lower()
    vw = vw if vw in ("vansh", "both") else "param"
    person = d.get("person") or old.get("person") or settings.get("person") or (sj or {}).get("person")
    person = None if person in ("None", "") else person
    pfs = d.get("prefill_s", settings.get("prefill_s", (old.get("prefill") or {}).get("s")))
    if sj and isinstance(sj.get("prefill"), dict) and int(sj["prefill"].get("frames") or 0) > 0:
        pfs = round((int(sj["prefill"]["frames"]) - 2) * 0.08, 2)          # P's effective prefill: N = 2 x round(s / 0.16) + 2 code frames
    elif sj:
        pfs = 0.0
    rq = dict(old, run_id=sid, model="livedup", session=sid, client="genweb-livedup", ui=settings, params=_engine_params(settings, sj),
              label=str(d.get("label") or old.get("label") or "live call")[:120], video_weights=vw, person=person, stream=True,
              prefill={"s": float(pfs or 0.0), "person": person, "src": settings.get("prefill_src")},
              url_query=str(d.get("url_query") or old.get("url_query") or "")[:4000], gt=bool(d.get("gt", old.get("gt", False))))
    rq.setdefault("created", time.time())
    if vw != "param":
        rq["vansh_steps"] = 50 if str(d.get("vansh_steps") or old.get("vansh_steps")) == "50" else 20
    else:
        rq.pop("vansh_steps", None)
    if sj:
        rq["seed"] = sj.get("seed")
        rq["session_dir"] = _session_dir(sid)
    jwrite(f"{rd}/request.json", rq)
    st = jread(f"{rd}/status.json", {}) or {}
    if st.get("state") not in ("done", "error", "importing"):
        jwrite(f"{rd}/status.json", {"state": "live", "progress": 0, "msg": "live call in progress", "model": "livedup", "worker": "livedup",
                                     "t_start": rq["created"]})
    return rq


async def start(req):
    d = await req.json()
    sid = str(d.get("session") or "")
    if not C["RID"].match(sid):
        return web.json_response({"error": f"bad session id {sid[:40]!r}"}, status=400)
    with LOCK:
        await asyncio.to_thread(_register, sid, d)
    print(f"[genweb] livedup start {sid}", flush=True)
    return web.json_response({"run_id": sid})


async def stop(req):
    d = await req.json()
    sid = str(d.get("session") or "")
    if not C["RID"].match(sid):
        return web.json_response({"error": "bad session id"}, status=400)
    rd = f"{C['RUNS']}/{sid}"
    st = C["jread"](f"{rd}/status.json", {}) or {}
    if st.get("state") == "live":
        C["jwrite"](f"{rd}/status.json", dict(st, msg="call ended; saving the recording", t_stop=time.time()))
    return web.json_response({"run_id": sid, "state": st.get("state") or "unknown"})


async def runstate(req):
    sid = req.match_info["sid"]
    if not C["RID"].match(sid):
        raise web.HTTPNotFound()
    st = C["jread"](f"{C['RUNS']}/{sid}/status.json", {}) or {}
    return web.json_response({"run_id": sid, "state": st.get("state"), "msg": st.get("msg"), "headline": st.get("headline"), "error": st.get("error")},
                             headers={"Cache-Control": "no-store"})


def _queue_render(rid):
    """= the workers' hand-off: the run's feed -> its render queue (Param: render/jobs; Vansh: vdir(vsteps)); Both: Param + compare.json"""
    RUNS, ROOT, jread, jwrite = C["RUNS"], C["ROOT"], C["jread"], C["jwrite"]
    rq = jread(f"{RUNS}/{rid}/request.json", {}) or {}
    jd = C["vdir"](C["vsteps"](rid, rq)) if rq.get("video_weights") == "vansh" else f"{ROOT}/render/jobs"
    os.makedirs(jd, exist_ok=True)
    if not os.path.exists(f"{jd}/{rid}.feed.jsonl") and not os.path.exists(f"{C['ROUT']}/{rid}.mp4"):
        with open(f"{RUNS}/{rid}/feed.jsonl", "rb") as fi, open(f"{jd}/{rid}.feed.jsonl.tmp", "wb") as fo:
            fo.write(fi.read())
        os.replace(f"{jd}/{rid}.feed.jsonl.tmp", f"{jd}/{rid}.feed.jsonl")
    if rq.get("video_weights") == "both" and not os.path.exists(f"{RUNS}/{rid}/compare.json") and not os.path.exists(f"{C['ROUT']}/{rid}.vansh.mp4"):
        jwrite(f"{RUNS}/{rid}/compare.json", {"other": "vansh", "at": time.time(), "steps": rq.get("vansh_steps") or 20, "why": "video weights: both"})


def _import(sid, sd):
    RUNS, jread, jwrite = C["RUNS"], C["jread"], C["jwrite"]
    rd = f"{RUNS}/{sid}"
    sj = jread(f"{sd}/session.json", {}) or {}
    with LOCK:
        _register(sid, {}, sj)
    jwrite(f"{rd}/status.json", dict(jread(f"{rd}/status.json", {}) or {}, state="importing", msg="saving the recording"))
    r = subprocess.run(["nice", "-n", "10", PY, f"{HERE}/livedup_import.py", sd, rd], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        mt = os.path.getmtime(f"{sd}/session.json")
        n = FAILED.get(sid, (mt, 0))[1] + 1
        FAILED[sid] = (mt, n)
        err = (r.stderr or r.stdout or "")[-600:]
        print(f"[genweb] livedup import {sid} failed ({n}): {err}", flush=True)
        jwrite(f"{rd}/status.json", {"state": "error" if n >= 3 else "importing", "error": f"recording import failed: {err[-300:]}", "model": "livedup",
                                     "worker": "livedup", "msg": "recording import failed" if n >= 3 else "retrying the recording import",
                                     "t_end": time.time()})
        return
    print((r.stdout or "").strip()[-300:], flush=True)
    _queue_render(sid)


def importer():
    while True:
        try:
            RUNS, jread = C["RUNS"], C["jread"]
            todo = []
            sids = sorted(os.listdir(P_SESS), reverse=True)[:200] if os.path.isdir(P_SESS) else []
            for sid in sids:
                if C["RID"].match(sid):
                    todo.append((sid, f"{P_SESS}/{sid}"))
            for rid in sorted(os.listdir(RUNS), reverse=True)[:300]:     # runs the page registered on another instance (fallback)
                if C["RID"].match(rid) and rid not in sids and os.path.exists(f"{RUNS}/{rid}/request.json"):
                    rq = jread(f"{RUNS}/{rid}/request.json", {}) or {}
                    if rq.get("model") == "livedup" and not rq.get("hidden"):   # 10-26: never agent test calls
                        sd = _session_dir(rid)
                        if sd:
                            todo.append((rid, sd))
                        elif (jread(f"{RUNS}/{rid}/status.json", {}) or {}).get("state") == "live" and time.time() - float(rq.get("created") or 0) > 2700:
                            C["jwrite"](f"{RUNS}/{rid}/status.json", {"state": "error", "error": "the live backend never saved this call", "model": "livedup"})
            for sid, sd in todo:
                if not os.path.exists(f"{sd}/session.json") or not os.path.exists(f"{sd}/session.npz"):
                    if sd.startswith(P_SESS) and not os.path.exists(f"{RUNS}/{sid}/request.json") and os.path.exists(f"{sd}/feed.jsonl"):
                        with LOCK:
                            _register(sid, {})                           # a call in progress the page has not registered (yet)
                    continue
                if time.time() - os.path.getmtime(f"{sd}/session.json") < 2:
                    continue
                st = (jread(f"{RUNS}/{sid}/status.json", {}) or {}).get("state")
                if st in ("done", "error"):
                    continue
                hit = FAILED.get(sid)
                if hit and hit[1] >= 3 and hit[0] == os.path.getmtime(f"{sd}/session.json"):
                    continue
                _import(sid, sd)
        except Exception as e:  # noqa: BLE001
            print(f"[genweb] livedup importer: {e!r}"[:400], flush=True)
        time.sleep(3.0)


def setup(app, g):
    C.update({k: g[k] for k in ("RUNS", "ROUT", "ROOT", "RID", "jread", "jwrite", "vdir", "vsteps")})
    app.router.add_get("/api/livedup/config", config)
    app.router.add_post("/api/livedup/start", start)
    app.router.add_post("/api/livedup/stop", stop)
    app.router.add_get("/api/livedup/run/{sid}", runstate)
    threading.Thread(target=importer, daemon=True).start()
