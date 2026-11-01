"""10-26 user (via main): the playground's HISTORY tab (every past run and batch, newest first), built from the run dirs on disk on every
request, so it survives server restarts. Registered by server.py: history_api.setup(app, globals()).

  GET /api/history          -> {"groups": [...]} (fields of the first /api/history kept: batch, n, done, failed, status, thumb = the first exact-render
                               mp4, seeds, created, first, runs, ...; plus kind, poster = the jpg below, key params, states): one row per batch (request.json "batch".id), per older n_samples press (runs from before
                               batch ids: consecutive runs with the same model + identical ui, ui.n_samples > 1, queued < 60 s apart, at most
                               n_samples of them: id "L<first run id>") or per single run; newest first; per-run summaries cached by mtimes
  legacy_runs("L<rid>")     -> that older press's run ids (server.py's /api/batch/<id> + zip accept these ids through it)
  GET /api/thumb/<rid>.jpg  -> a small poster frame of the run's exact render (render/out/<rid>.mp4, else the live recording), made with
                               ffmpeg on demand (t = 0.5 s, before the prefill flash) and cached in render/thumbs/ (remade when the video changes)
"""
import asyncio
import time
import json
import os
import subprocess

from aiohttp import web

C = {}                       # server globals (RUNS, ROUT, RID, BID, FF, run_summary, exact_src, card_state, jread)
CACHE = {}                   # rid -> (mtime key, row)
THUMB_SEM = []               # the ffmpeg limiter (made on first use, inside the server loop)
KEYS = ("text_temp", "text_topk", "min_p", "freq_pen", "cb0_topk", "light_topk", "pose_temp", "pose_topk", "pose_light_topk", "pose_cfg_on",
        "pose_cfg_s", "dup_mode")


def _mt(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0


POSE = {}                    # run dir -> (pose_stats.json mtime, stats)
GEST = {}                    # run dir -> (gesture_stats.json mtime, fields)   (10-26 gesturemetric: gesture timing kappa)


def gesture_stats(d):
    """10-26 main/gesturemetric: per-run gesture TIMING score written by gesturemetric's own CPU job (ppm/agents1026/gesturemetric/gm_run.py
    -> runs/<id>/gesture_stats.json, atomic). Read only, cached by mtime. timing_kappa = gesture strokes within +-0.2 s of B's own word onsets
    vs chance (0 = unrelated; real Aloe 18 s clips ~0.28); timing_kappa_real = its ref.clip_real.words.kappa. None until the file exists."""
    gm = _mt(f"{d}/gesture_stats.json")
    if not gm:
        return None
    hit = GEST.get(d)
    if not (hit and hit[0] == gm):
        try:
            with open(f"{d}/gesture_stats.json") as f:                 # one JSON object; tolerate stray log lines before it (gm_run's
                g = json.loads([ln for ln in f.read().splitlines() if ln.lstrip().startswith("{")][-1])   # tokenizer prints to stdout)
            rw = (((g.get("ref") or {}).get("clip_real") or {}).get("words") or {})
            g = {"timing_kappa": g.get("gesture_timing_kappa"), "timing_loud": g.get("gesture_timing_loud"),
                 "strokes_per_min": g.get("strokes_per_min"), "timing_kappa_real": rw.get("kappa"),
                 "timing_kappa_real_p10_p90": rw.get("kappa_p10_p90")}
        except (OSError, ValueError, AttributeError, IndexError):
            g = None
        GEST[d] = hit = (gm, g)
    return hit[1]


def pose_stats(d):
    """the run's pose stats (posefix's pose_stats.json, below) + gesturemetric's timing kappa fields (gesture_stats.json) when present."""
    p = _pose_stats(d)
    g = gesture_stats(d)
    return dict(p or {}, **{k: v for k, v in g.items() if v is not None}) if g else p


def _pose_stats(d):
    """10-26 user: the run's pose stats over the GENERATED tokens only (>= the N prefill frames; duplex = B's pose): cb0 repeat %, out-of-frame %,
    keypoint confidence. Computed by posefix's scorer in its own job (genweb/pose_scorer.py -> runs/<id>/pose_stats.json); this only reads the
    file (cached by its mtime). Not scored yet (or tokens changed since) -> cb0 repeat % = 100 - the worker's own 'pose cb0 changes on X%'
    (summary.json gen_cb0_change, the same definition), marked pending."""
    tm = _mt(f"{d}/tokens.npz")
    if not tm:
        return None
    pm = _mt(f"{d}/pose_stats.json")
    hit = POSE.get(d)
    if not (hit and hit[0] == pm):
        try:
            with open(f"{d}/pose_stats.json") as f:
                ps = json.load(f)
        except (OSError, ValueError):
            ps = None
        POSE[d] = hit = (pm, ps)
    ps = hit[1]
    if ps and ps.get("tokens_mtime") == tm and ps.get("cb0_repeat") is not None:
        return {k: ps.get(k) for k in ("cb0_repeat", "oof", "conf", "lowconf", "prefill_frames", "gen_frames", "gestures_per_min",
                                        "gestures_per_min_left", "gestures_per_min_right", "gestures", "hands_moving_pct", "hand_speed_p50",
                                        "hand_speed_p95", "hand_seconds")}
    try:
        with open(f"{d}/summary.json") as f:
            sm = json.load(f)
        chg = sm.get("gen_cb0_change")
        return {"cb0_repeat": round(100 * (1 - chg), 1), "prefill_frames": sm.get("pose_prefill_frames", sm.get("prefill_frames")),
                "pending": True} if chg is not None else None
    except (OSError, ValueError):
        return None


# ---- 10-26 user: honest ETAs in the run status text. A background thread rebuilds a snapshot every 3 s (never in a request):
# generation queue per model (READY runs, oldest first) with a start estimate = now + (frames still to generate in the runs ahead + the
# running runs' remaining frames) x the measured ms/frame (median of the model's last 5 finished runs) / the live worker count (running runs
# whose worker job is alive in squeue; never < 1), and the render queues (render/jobs*: pending feeds oldest first). Render progress rate:
# observed from the render service's progress_s each time a run's status is read.
ETA = {"t": 0, "q": {}, "rq": {}}
ETA_T, RATE, SQ = [], {}, {"t": 0, "ids": set()}
ENDT = {}                                                               # rid -> first time its render progress reached the end (encode stage)
# 10-26: the render's last stages, timed from the recent outputs (refreshed every 60 s): render_service encodes the side-by-side (double
# width) video, then the main mp4 (= its done signal); genweb's flasher then burns the flash + beep. Measured on 12 runs: encode + sxs =
# 0.32 s per video second (main encode alone 0.13, sxs ~1.5x that), flash 0.24 s per video second (+ ~2 s). K: seconds per video second.
K = {"t": 0, "enc": 0.33, "flash": 0.24}


def _stage_k():
    import glob
    import statistics
    import time
    ROUT = C["ROUT"]
    fl = sorted(glob.glob(f"{ROUT}/2*.flash.mp4"), key=_mt)[-12:]
    enc, fla = [], []
    for f in fl:
        rid = os.path.basename(f).split(".")[0]
        m, x = f"{ROUT}/{rid}.mp4", f"{ROUT}/{rid}_sxs.mp4"
        sec = (C["jread"](f"{ROUT}/{rid}.json", {}) or {}).get("seconds")
        if not sec or not os.path.exists(m):
            continue
        fla.append((_mt(f) - _mt(m)) / sec)
        if os.path.exists(x):
            enc.append(2.5 * (_mt(m) - _mt(x)) / sec)                   # encode + sxs = 2.5 x the main encode (measured 2.4-2.6)
    if enc:
        K["enc"] = statistics.median(enc)
    if fla:
        K["flash"] = statistics.median(fla)
    K["t"] = time.time()


def _pt_hm(ts):
    from zoneinfo import ZoneInfo
    import datetime as dt
    return dt.datetime.fromtimestamp(ts, ZoneInfo("America/Los_Angeles")).strftime("%H:%M")


def _jobs_alive():
    import subprocess as sp
    import time
    if time.time() - SQ["t"] > 30:
        try:
            out = sp.run(["squeue", "-h", "-u", os.environ.get("USER", "amartyavarma"), "-o", "%i"], capture_output=True, text=True, timeout=20).stdout
            SQ["ids"], SQ["t"] = set(out.split()), time.time()
        except Exception:  # noqa: BLE001
            SQ["t"] = time.time()
    return SQ["ids"]


def _frames(req):
    pf = (req.get("prefill") or {}).get("s") or 0
    return (float(req.get("clip_s") or 20) + float(pf or 0)) / 0.08 + 2


def _eta_snapshot():
    import re as _re
    import statistics
    import time
    RUNS, ROUT = C["RUNS"], C["ROUT"]
    now, alive = time.time(), _jobs_alive()
    per = {}                                                            # model -> {"q": [(created, rid, frames)], "rem": frames, "w": n, "ms": [(t_end, ms)]}
    for rid in os.listdir(RUNS):
        if not C["RID"].match(rid):
            continue
        d = f"{RUNS}/{rid}"
        st = C["jread"](f"{d}/status.json", {}) or {}
        state = st.get("state")
        if state not in ("queued", "running", "claimed", "done") and not os.path.exists(f"{d}/READY"):
            continue
        req = C["jread"](f"{d}/request.json", {}) or {}
        m = per.setdefault(req.get("model") or "monoplex", {"q": [], "rem": 0.0, "w": 0, "ms": []})
        if os.path.exists(f"{d}/READY") and state in (None, "queued"):
            if not req.get("hidden"):                                   # agent test runs are claimed last: never ahead of a user run
                m["q"].append((req.get("created") or 0, rid, _frames(req)))
        elif state in ("running", "claimed"):
            tot, done = st.get("frames_total") or _frames(req), st.get("frames_done") or 0
            m["rem"] += max(0.0, float(tot) - float(done))
            j = _re.search(r"(\d{5,})", str(st.get("worker") or ""))
            if j and j.group(1) in alive:
                m["w"] += 1
        elif state == "done" and st.get("ms_per_frame"):
            m["ms"].append((st.get("t_end") or 0, float(st["ms_per_frame"])))
    q = {}
    for m in per.values():
        ms = [x[1] for x in sorted(m["ms"])[-5:]]
        msf = statistics.median(ms) if ms else 70.0
        w, cum = max(1, m["w"]), m["rem"]
        for i, (_, rid, fr) in enumerate(sorted(m["q"])):
            q[rid] = (i, now + cum * msf / 1000.0 / w)
            cum += fr
    rq = {}
    try:
        outs = set(os.listdir(ROUT))
    except OSError:
        outs = set()
    rdir = os.path.dirname(ROUT)
    for jd in ("jobs", "jobs_vansh", "jobs_vansh50"):                   # only the queues the renderers consume (never parked / test dirs)
        try:
            fs = [f for f in os.listdir(f"{rdir}/{jd}") if f.endswith(".feed.jsonl")]
        except OSError:
            continue
        pend = []
        for f in fs:
            n = f[:-len(".feed.jsonl")]
            if f"{n}.mp4" in outs:
                continue
            if C["RID"].match(n) and (C["jread"](f"{C['RUNS']}/{n}/request.json", {}) or {}).get("hidden"):
                continue                                                # 10-26: agent test renders are not counted ahead
            pend.append((_mt(f"{rdir}/{jd}/{f}"), n))
        for i, (_, n) in enumerate(sorted(pend)):
            rq.setdefault(n, i)
    ETA.update(t=now, q=q, rq=rq)
    if now - K["t"] > 60:
        _stage_k()


def _eta_loop():
    import time
    while True:
        try:
            _eta_snapshot()
        except Exception as e:  # noqa: BLE001
            print(f"[genweb] eta: {e!r}"[:300], flush=True)
        time.sleep(3)


def eta_text(rid, o):
    """the honest status line for a queued run / a render in progress / a queued render, else None (reads the snapshot only)"""
    import time
    if not ETA_T:
        import threading
        ETA_T.append(threading.Thread(target=_eta_loop, daemon=True))
        ETA_T[0].start()
    st, r = o.get("state"), o.get("render") or {}
    if st == "done" and r.get("src") and not r.get("flash") and r.get("switch_s") is not None and not r.get("approx"):
        m = f"{C['ROUT']}/{rid}.mp4"                                    # the exact mp4 is out; genweb's flasher is burning the flash + beep
        if os.path.exists(m):
            sec = float(r.get("seconds") or 0)
            left = max(1, round(2 + K["flash"] * sec - (time.time() - _mt(m))))
            r["stage"] = f"adding the flash + beep · ~{left} s"
            return r["stage"]
    if st in (None, "queued", "unknown") and rid in ETA["q"]:
        n, t0 = ETA["q"][rid]
        return f"queued · {n} ahead · starts ~{_pt_hm(max(t0, time.time()))} PT"
    if st != "done" or r.get("src"):
        return None
    if r.get("state") == "encoding":                                   # 10-26 videort: encodes while rendering; only a ~1 s tail is left
        r["stage"] = "writing the video file · ~1 s"
        return r["stage"]
    if r.get("state") == "rendering":
        ps, sec, now = float(r.get("progress_s") or 0), float(r.get("seconds") or 0), time.time()
        if sec and ps >= sec - 1:                                       # all frames made (a renderer from before encode-while-rendering)
            t_end = ENDT.setdefault(rid, now)
            left = max(1, round(K["enc"] * sec - (now - t_end)))
            r["stage"] = f"writing the video file (encode + side-by-side) · ~{left} s"
            return r["stage"]
        obs = RATE.setdefault(rid, [])
        if not obs or obs[-1][1] != ps:
            obs.append((now, ps))
        obs[:] = [x for x in obs if now - x[0] < 120][-20:]
        txt = f"making the avatar video · {ps:.0f} / {sec:.0f} s"
        if len(obs) >= 2 and obs[-1][1] > obs[0][1] and sec > ps:
            rate = (obs[-1][1] - obs[0][1]) / max(1e-3, obs[-1][0] - obs[0][0])
            r["done_at"] = _pt_hm(now + (sec - ps) / rate) + " PT"
            txt += f" · done ~{r['done_at']}"
        return txt
    if r.get("state") != "fail":
        n = ETA["rq"].get(rid)
        return f"render queued · {n} ahead" if n is not None else None
    return None


def run_row(rid):
    RUNS, ROUT = C["RUNS"], C["ROUT"]
    d = f"{RUNS}/{rid}"
    key = (_mt(f"{d}/status.json"), _mt(f"{d}/request.json"), _mt(d), _mt(f"{ROUT}/{rid}.json"), _mt(f"{ROUT}/{rid}.mp4"),
           _mt(f"{ROUT}/{rid}.flash.mp4"), _mt(f"{d}/avatar.mp4"))
    hit = CACHE.get(rid)
    if hit and hit[0] == key:
        hit[1]["pose"] = pose_stats(d)                                  # scored in the background after the row was cached
        return hit[1]
    o = C["run_summary"](rid)
    src, _ = C["exact_src"](rid)
    state, note = C["card_state"](o, src)
    req = C["jread"](f"{d}/request.json", {}) or {}
    tsrc = f"{ROUT}/{rid}.mp4" if os.path.exists(f"{ROUT}/{rid}.mp4") else (f"{d}/avatar.mp4" if os.path.exists(f"{d}/avatar.mp4") else None)
    b = req.get("batch") if isinstance(req.get("batch"), dict) else {}
    dm = ((req.get("params") or {}).get("dup_mode") or req.get("dup_mode") or (req.get("ui") or {}).get("dup_mode")
          or ("replay" if (o.get("model") or "monoplex") == "duplex" else None))      # 10-26 dupui: duplex generate mode; old duplex runs = replay
    row = {"run_id": rid, "created": o.get("created") or 0, "model": o.get("model") or "monoplex", "seed": o.get("seed"), "state": state,
           "note": note, "headline": o.get("headline") or "", "label": o.get("label") or "", "person": o.get("person"), "voice": o.get("voice"),
           "clip_s": o.get("clip_s"), "prefill": o.get("prefill") or {}, "stream": o.get("stream"), "ui": req.get("ui") or {},
           "params": req.get("params") or {}, "conv": req.get("conv"), "side": req.get("side"), "video_weights": req.get("video_weights") or "param", "vansh_steps": req.get("vansh_steps") if req.get("vansh_steps") in (20, 50) else (50 if (req.get("video_weights") or "param") != "param" else None),   # 10-26: no field = 50 (old)
           "batch": str(b.get("id")) if C["BID"].match(str(b.get("id") or "")) else None, "k": b.get("k", 0),
           "thumb": f"api/thumb/{rid}.jpg?v={int(_mt(tsrc))}" if tsrc else None, "dup_mode": dm, "pose": o.get("pose"),
           "hidden": bool(req.get("hidden"))}
    CACHE[rid] = (key, row)
    return row


def group_rows(rows):
    """rows newest first -> groups newest first"""
    groups, by = [], {}
    legacy = None                                                       # the open older-press group (rows walked oldest first)
    for r in sorted(rows, key=lambda r: (r["created"], r["run_id"])):
        if r.get("hidden"):                                             # 10-26: agent test runs are never listed
            continue
        if r["batch"]:
            g = by.get(r["batch"])
            if g is None:
                g = by[r["batch"]] = {"id": r["batch"], "kind": "batch", "rows": []}
                groups.append(g)
            g["rows"].append(r)
            legacy = None
            continue
        ui = r["ui"] or {}
        n = int(ui.get("n_samples") or 1) if str(ui.get("n_samples") or "1").isdigit() else 1
        sig = json.dumps([r["model"], ui], sort_keys=True)
        if (legacy and n > 1 and legacy["sig"] == sig and r["created"] - legacy["rows"][-1]["created"] < 60 and len(legacy["rows"]) < n):
            legacy["rows"].append(r)
            continue
        g = {"id": ("L" if n > 1 else "") + r["run_id"], "kind": "legacy" if n > 1 else "single", "rows": [r], "sig": sig}
        groups.append(g)
        legacy = g if n > 1 else None
    out = []
    for g in groups:
        rs = sorted(g["rows"], key=lambda r: (r["k"] or 0, r["created"], r["run_id"]))
        f = rs[0]
        st = [r["state"] for r in rs]
        if g["kind"] == "legacy" and len(rs) == 1:
            g["kind"], g["id"] = "single", f["run_id"]
        status = next((s for s in ("generating", "rendering", "queued") if s in st), "failed" if "failed" in st and "done" not in st else "done")
        pf = f["prefill"] or {}
        keyp = {k: f["params"][k] for k in KEYS if k in (f["params"] or {})}
        if f.get("dup_mode"):
            keyp["dup_mode"] = f["dup_mode"]
        ps = [r["pose"] for r in rs if r.get("pose") and r["pose"].get("cb0_repeat") is not None]
        mean = lambda k, nd: (round(sum(p[k] for p in ps if p.get(k) is not None) / max(1, sum(1 for p in ps if p.get(k) is not None)), nd)
                              if any(p.get(k) is not None for p in ps) else None)
        pose = {"cb0_repeat": mean("cb0_repeat", 1), "oof": mean("oof", 1), "conf": mean("conf", 2), "n": len(ps),
                "gestures_per_min": mean("gestures_per_min", 1), "hands_moving_pct": mean("hands_moving_pct", 1),
                "min": min(p["cb0_repeat"] for p in ps), "max": max(p["cb0_repeat"] for p in ps), "prefill_frames": ps[0].get("prefill_frames"),
                "pending": any(p.get("pending") for p in ps)} if ps else None
        if pose and C.get("pose_refs"):
            pose["ref"] = C["pose_refs"](f["person"])                       # 10-26 aloedemo: gestures refs (this actor real · Aloe)
        out.append({"id": g["id"], "kind": g["kind"], "created": min(r["created"] for r in rs), "n": len(rs), "done": st.count("done"),
                    "failed": st.count("failed"), "status": status, "runs": [r["run_id"] for r in rs], "seeds": [r["seed"] for r in rs],
                    "states": st, "first": f["run_id"], "model": f["model"], "person": f["person"], "voice": f["voice"],
                    "label": f["label"], "clip_s": f["clip_s"], "prefill_s": pf.get("s"), "prefill_src": pf.get("src"), "stream": f["stream"],
                    "conv": f["conv"], "side": f["side"], "headline": next((r["headline"] for r in rs if r["headline"]), ""),
                    "poster": next((r["thumb"] for r in rs if r["thumb"] and r["state"] == "done"), None) or next((r["thumb"] for r in rs if r["thumb"]), None),
                    "thumb": next((C["exact_src"](r["run_id"])[0] for r in rs if C["exact_src"](r["run_id"])[0]), None),   # = the first /api/history: an mp4 (the batch list)
                    "key": keyp, "ui": f["ui"], "params": f["params"], "batch": g["kind"] == "batch", "dup_mode": f.get("dup_mode"), "pose": pose, "video_weights": f.get("video_weights") or "param", "vansh_steps": f.get("vansh_steps"),
                    "video": next((C["exact_src"](r["run_id"])[0] for r in rs if r["state"] == "done"), None)})
    out.sort(key=lambda g: g["created"], reverse=True)
    return out


def legacy_runs(gid):
    """the run ids of an older n_samples press "L<first run id>" (the same grouping as the History rows), [] if there is no such group"""
    try:
        ids = [r for r in os.listdir(C["RUNS"]) if C["RID"].match(r)]
    except OSError:
        return []
    rows = []
    for r in ids:
        try:
            rows.append(run_row(r))
        except Exception:  # noqa: BLE001
            pass
    return next((g["runs"] for g in group_rows(rows) if g["id"] == gid and g["kind"] == "legacy"), [])


HCACHE = {}


async def hist(req):
    try:
        ids = [r for r in os.listdir(C["RUNS"]) if C["RID"].match(r)]
    except OSError:
        ids = []
    def rows_of():
        out = []
        for r in ids:
            try:
                out.append(run_row(r))
            except Exception as e:  # noqa: BLE001
                print(f"[genweb] history: {r}: {e!r}"[:300], flush=True)
        return out
    hit = HCACHE.get("rows")                                                  # 10-26 (page hangs): one scan per 2 s shared by every poller
    if hit and time.time() - hit[0] < 2.0:
        rows = hit[1]
    else:
        rows = await asyncio.to_thread(rows_of)
        HCACHE["rows"] = (time.time(), rows)
    return web.json_response({"groups": group_rows(rows)}, headers={"Cache-Control": "no-store"})


async def thumb(req):
    rid = req.match_info["rid"]
    RUNS, ROUT = C["RUNS"], C["ROUT"]
    base, _, var = rid.partition(".")                                  # 10-26: <rid>.vansh / <rid>.param = the other-weights render (Both / compare)
    if not C["RID"].match(base) or var not in ("", "vansh", "param"):
        raise web.HTTPNotFound()
    src = f"{ROUT}/{rid}.mp4" if os.path.exists(f"{ROUT}/{rid}.mp4") else (f"{RUNS}/{rid}/avatar.mp4" if not var and os.path.exists(f"{RUNS}/{rid}/avatar.mp4") else None)
    if not src:
        raise web.HTTPNotFound()
    td = f"{os.path.dirname(ROUT)}/thumbs"
    out = f"{td}/{rid}.jpg"
    if _mt(out) < _mt(src):
        os.makedirs(td, exist_ok=True)
        if not THUMB_SEM:
            THUMB_SEM.append(asyncio.Semaphore(3))
        async with THUMB_SEM[0]:
            r = await asyncio.to_thread(subprocess.run, [C["FF"], "-y", "-loglevel", "error", "-ss", "0.5", "-i", src, "-frames:v", "1",
                                                         "-vf", "scale=128:-2", "-q:v", "5", out + ".tmp.jpg"], capture_output=True, timeout=60)
        if r.returncode != 0 or not os.path.exists(out + ".tmp.jpg"):
            raise web.HTTPNotFound()
        os.replace(out + ".tmp.jpg", out)
    return web.FileResponse(out, headers={"Cache-Control": "max-age=86400"})


def setup(app, g):
    C.update({k: g[k] for k in ("RUNS", "ROUT", "RID", "BID", "FF", "run_summary", "exact_src", "card_state", "jread", "pose_refs") if k in g})
    app.router.add_get("/api/history", hist)
    app.router.add_get("/api/thumb/{rid}.jpg", thumb)
