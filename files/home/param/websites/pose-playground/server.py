"""MONOPLEX PLAYGROUND web server (10-25, user: "a website where I can easily toggle inference parameters and run generations").

The page (index.html) renders every engine knob from the worker's schema, queues runs and shows results. No GPU here: posefix's warm
workers do the generations. Contract with the workers = files under ROOT (no HTTP between us):
  SCHEMA.json   worker-written: rows {name, label, group, type float|int|bool|enum|text, default, range, step, choices, hint}
  OPTIONS.json  worker-written: {voices:[{id,label}], persons:[...], max_parallel, ...} (pickers / notes)
  runs/<id>/request.json + READY   written here on Generate: {run_id, created, params:{<schema name>: value}, voice:{kind, id | wav}}
  runs/<id>/CLAIMED.<worker>       the worker claims a run by renaming READY
  runs/<id>/status.json            worker: {state queued|running|done|error, progress, msg, headline, t_start, t_end, error}
  runs/<id>/{b.wav,a.wav,words.json,avatar.mp4,gt.mp4} (+ s<k>/ per sample)   worker outputs, served as they appear
  uploads/<hex>.wav               voice clips uploaded from the page (the worker does LUFS + Mimi)
Plain URL, no auth (as the live demo).   python server.py --port 9205
"""
import argparse
import asyncio
import glob
import io
import shutil
import subprocess
import threading
import json
import os
CANOPY_ROOT = os.environ.get("CANOPY_ROOT") or os.path.expanduser("~")   # cluster home holding ppm/, venvs/, models/ (see README: paths)
HOME_DIR = os.path.expanduser("~")
import re
import secrets
import time
import zipfile
import history_api                                                    # 10-26: History + pose stats (history_api.py)

from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("GENWEB_ROOT", HERE)
RUNS, UPL, ROUT = f"{ROOT}/runs", f"{ROOT}/uploads", f"{ROOT}/render/out"
RSEQ = f"{ROOT}/render/out_seq"          # 10-26: sequential-sampler re-renders of runs rendered before the switch
RSEQ2 = f"{ROOT}/render/out_seq2"        # 10-26: re-renders after the SeqEngine ref-seed fix
RJV = f"{ROOT}/render/jobs_vansh"        # 10-26 user: Vansh-weights render queue (videort's Vansh renderers); outputs still land in render/out
RJV50 = f"{ROOT}/render/jobs_vansh50"    # 10-26 user ("20 steps not 50"): jobs_vansh = Vansh 20 steps (default), jobs_vansh50 = 50 steps
def vsteps(rid, rq):
    """Vansh sampler steps of a run's Vansh render: the request's vansh_steps, else the compare sidecar, else 50 (renders made before 10-26 06:40)"""
    s = rq.get("vansh_steps") if isinstance(rq, dict) else None
    if s not in (20, 50):
        s = (jread(f"{RUNS}/{rid}/vansh_steps.json", {}) or {}).get("steps")
    return s if s in (20, 50) else 50
def vdir(steps):
    return RJV if steps == 20 else RJV50
VANSH_LIVE = f"{ROOT}/VANSH_LIVE"        # 10-26: exists once videort serves Vansh weights live; until then Vansh runs render only
RID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
BID = re.compile(r"^B[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")   # 10-26: one Generate press = one batch (request.json "batch": {id, n, k})
OUTS = ("avatar.mp4", "gt.mp4", "b.wav", "a.wav", "words.json", "headline.txt", "stick.mp4")
MAX_UPLOAD = 25 << 20
SERVER_V = "v2026.10.26-23"               # 10-26: this deploy's CHANGELOG.json version (GET /api/changelog "live")
LOADED = {"server_sha": __import__("hashlib").sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:8],
          "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}   # what THIS process loaded (version panel)


FF = os.environ.get("GENWEB_FFMPEG", "ffmpeg")
LIVE = os.environ.get("GENWEB_LIVE", CANOPY_ROOT + "/ppm/aloe-posedup-live")
RENDERERS = os.environ.get("GENWEB_RENDERERS", "param,vansh").split(",")
BATCH_STREAM = os.environ.get("GENWEB_BATCH_STREAM", "1") == "1"


def switch_s(req):
    """time of the prefill -> generated switch in the avatar video (the real-video prefill = whole 0.16 s latents), or None."""
    s = float((req.get("prefill") or {}).get("s") or 0)
    return round(round(s / 0.16) * 0.16, 2) if s > 0 else None


def reaper():
    """10-26 (a flex reshuffle killed a worker mid-run; the page froze at 125/900): every 10 s, a RUNNING run whose status is >30 s old and whose
    worker's Slurm job is gone (worker id ends in the job id, e.g. mono30601 / flexmono30643 / dup30602), or whose status is >15 min old, is
    re-queued (claim dropped, READY back) and shows 'worker restarted, re-queued'."""
    while True:
        try:
            now = time.time()
            for rid in sorted(os.listdir(RUNS), reverse=True)[:300]:
                rd = f"{RUNS}/{rid}"
                sp = f"{rd}/status.json"
                try:
                    age = now - os.path.getmtime(sp)
                except OSError:
                    continue
                if age < 30:
                    continue
                st = jread(sp, {})
                if st.get("state") != "running":
                    continue
                wid = str(st.get("worker") or "")
                m = re.search(r"(\d{5})", wid)
                gone = False
                if m:
                    try:
                        out = subprocess.run(["squeue", "-h", "-j", m.group(1), "-o", "%T"], capture_output=True, text=True, timeout=8).stdout.strip()
                        gone = out not in ("RUNNING", "COMPLETING")
                    except (OSError, subprocess.SubprocessError):
                        gone = False
                if gone or age > 900:
                    for f in os.listdir(rd):
                        if f.startswith("CLAIMED."):
                            os.remove(f"{rd}/{f}")
                    jwrite(sp, {"state": "queued", "progress": 0, "msg": "worker restarted, re-queued", "requeued_from": wid, "t_requeue": now})
                    open(f"{rd}/READY", "w").close()
                    print(f"[genweb] reaper: {rid} re-queued (worker {wid} {'gone' if gone else 'silent 15 min'})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[genweb] reaper: {e}", flush=True)
        time.sleep(10)


# 10-26 (main): the flash + beep burn runs in its own job (genweb/flash_worker.py, 8 CPUs, up to 4 clips at once, newest first) with these
# exact functions; this server's flasher burns only while that job's heartbeat is stale (> 30 s). One burn per clip: render/out/<id>.flash.lock
# (O_EXCL; stale after 10 min).
FLASH_HB = f"{ROOT}/render/flash_worker.heartbeat"


def _mt0(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0


def flash_lock(rid):
    lk = f"{ROUT}/{rid}.flash.lock"
    for _ in range(2):
        try:
            os.close(os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            if time.time() - _mt0(lk) > 600:
                try:
                    os.remove(lk)
                except OSError:
                    pass
                continue
            return False
    return False


def flash_unlock(rid):
    try:
        os.remove(f"{ROUT}/{rid}.flash.lock")
    except OSError:
        pass


def burn_flash(src, out, t):
    """the white flash (3 frames) + 1 kHz beep at t s, burnt into src -> out (tmp + rename). Returns (ok, error text)."""
    ms = int(round(t * 1000))
    fc = (f"[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=white@1:t=fill:enable='between(t,{t:.3f},{t + 0.12:.3f})'[v];"
          f"[1:a]volume=0.25,adelay={ms}|{ms}[b];[0:a][b]amix=inputs=2:duration=first:normalize=0[a]")
    r = subprocess.run([FF, "-y", "-loglevel", "error", "-i", src, "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=0.12",
                        "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", out + ".tmp.mp4"], capture_output=True, timeout=300)
    if r.returncode == 0:
        os.replace(out + ".tmp.mp4", out)
        return True, ""
    return False, r.stderr.decode(errors="replace")[:300]


def flasher():
    """10-25 main: burn a white flash (3 frames) + a short beep at the prefill -> generated switch into each run's avatar mp4
    (<id>.flash.mp4 next to the render; the page serves it once it exists). CPU only, one clip at a time."""
    done = set()
    while True:
        try:
            # 10-26 (user via main / videort): rendering is Vansh's sequential sampler. New runs render straight into render/out; runs
            # rendered before the switch are re-rendered into render/out_seq/<id>.mp4, which then becomes the run's replay (the old engine's
            # mp4 / flash kept as <id>.mp4.pre-seq / <id>.flash.mp4.pre-seq) and gets its flash re-burnt.
            # 10-26 05:00 videort: SeqEngine's 9-ref pick used the wrong seed (fixed); everything SeqEngine rendered before the fix is re-rendered
            # into render/out_seq2/ (<rid>.mp4 and compare <rid>.vansh.mp4), promoted the same way (marker .pre-seq2; never undone by out_seq).
            for sd, mk in ((RSEQ, "pre-seq"), (RSEQ2, "pre-seq2")):
                for f in sorted(os.listdir(sd)) if os.path.isdir(sd) else []:
                    name = f[:-4]
                    rid = name[:-6] if name.endswith(".vansh") else name
                    if not f.endswith(".mp4") or not RID.match(rid) or os.path.exists(f"{ROUT}/{name}.mp4.{mk}") or os.path.exists(f"{ROUT}/{name}.mp4.pre-seq2"):
                        continue
                    if time.time() - os.path.getmtime(f"{sd}/{f}") < 5:
                        continue
                    if os.path.exists(f"{ROUT}/{name}.mp4"):
                        os.replace(f"{ROUT}/{name}.mp4", f"{ROUT}/{name}.mp4.{mk}")
                    else:
                        open(f"{ROUT}/{name}.mp4.{mk}", "w").close()
                    shutil.copyfile(f"{sd}/{f}", f"{ROUT}/{name}.mp4.tmp")
                    os.replace(f"{ROUT}/{name}.mp4.tmp", f"{ROUT}/{name}.mp4")
                    if name == rid and os.path.exists(f"{ROUT}/{rid}.flash.mp4"):
                        os.replace(f"{ROUT}/{rid}.flash.mp4", f"{ROUT}/{rid}.flash.mp4.{mk}")
                    done.discard(rid)
                    print(f"[genweb] {name}: replay = {sd.rsplit('/', 1)[-1]} re-render", flush=True)
            for rid in sorted(os.listdir(RUNS), reverse=True)[:400]:
                if rid in done or not RID.match(rid):
                    continue
                req = jread(f"{RUNS}/{rid}/request.json", {})
                st = jread(f"{RUNS}/{rid}/status.json", {})
                rj = f"{vdir(vsteps(rid, req)) if req.get('video_weights') == 'vansh' else ROOT + '/render/jobs'}/{rid}.feed.jsonl"   # 10-26: per video weights + Vansh steps
                # 10-26 main (exact maths): the replay is ALWAYS the render service's mp4 (from feed.jsonl = the run's exact full Mimi decode);
                # videort's live "record" carries the live stream's chunk-decoded audio, so it never counts as the replay
                have = os.path.exists(f"{ROUT}/{rid}.mp4")
                cl = f"{ROUT}/{rid}.claim"
                # 10-25 main: DURABLE render queue. A finished run without a video and without a queued / live render job is re-queued; a
                # claim older than 10 min with no video (its renderer died) is dropped so another renderer takes it.
                ovl = jread(f"{ROUT}/{rid}.json", {}) if not have else {}   # 10-27 overlap render: R is writing this run's mp4 right now
                if (str((ovl or {}).get("engine") or "").startswith("overlap") and (ovl or {}).get("state") == "rendering"
                        and time.time() - float((ovl or {}).get("at") or 0) < 1800):
                    continue                                            # the worker queues it itself if R's mp4 never shows up
                if st.get("state") == "done" and not have and os.path.exists(f"{RUNS}/{rid}/feed.jsonl"):
                    if os.path.exists(cl) and time.time() - os.path.getmtime(cl) > 600:
                        os.remove(cl)
                        jwrite(f"{ROUT}/{rid}.json", {"state": "queued", "note": "re-queued: the previous renderer stopped"})
                    if not os.path.exists(rj) and not os.path.exists(cl):
                        with open(f"{RUNS}/{rid}/feed.jsonl", "rb") as fi, open(rj + ".tmp", "wb") as fo:
                            fo.write(fi.read())
                        os.replace(rj + ".tmp", rj)
                        print(f"[genweb] re-queued render {rid}", flush=True)
                    continue                # videort's live backend recorded it: drop the duplicate render job if unclaimed
                if os.path.exists(f"{RUNS}/{rid}/avatar.mp4") and os.path.exists(rj) and not os.path.exists(f"{ROUT}/{rid}.claim") and not os.path.exists(f"{ROUT}/{rid}.mp4"):
                    os.remove(rj)
                t = switch_s(req)
                src = f"{ROUT}/{rid}.mp4"                                   # exact decode only (never the live recording)
                out = f"{ROUT}/{rid}.flash.mp4"
                if t is None:
                    if have:
                        done.add(rid)
                    continue
                if os.path.exists(out):
                    done.add(rid); continue
                if not have:
                    continue
                if not os.path.exists(src) or time.time() - os.path.getmtime(src) < 2:
                    continue
                if time.time() - _mt0(FLASH_HB) < 30:                       # 10-26: the flash job (flash_worker.py) burns; this is its fallback
                    continue
                if not flash_lock(rid):
                    continue
                try:
                    ok, err = burn_flash(src, out, t)
                finally:
                    flash_unlock(rid)
                if not ok:
                    print(f"[genweb] flash {rid} failed: {err}", flush=True)
                done.add(rid)
        except Exception as e:  # noqa: BLE001
            print(f"[genweb] flasher: {e}", flush=True)
        time.sleep(1.0)


def jread(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


RSLAST = {}


def rstatus(path):
    """10-26 (user: counter stuck at 0 while the json said 20 s): the renderer rewrites <id>.json in place several times a second, so a read
    can catch it half-written or briefly missing (it is replaced); that used to fall back to {} = progress 0. Retry, then keep the last good copy."""
    for _ in range(3):
        try:
            with open(path) as f:
                d = json.load(f)
            RSLAST[path] = d
            return d
        except (OSError, ValueError):                                   # incl. FileNotFoundError: the writer replaces the file (a brief gap)
            time.sleep(0.03)
    return RSLAST.get(path, {})


def jwrite(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def files_of(d, rel=""):
    """outputs present in a run dir (+ one level of sample sub-dirs s<k>/), with sizes; *.part = still being written."""
    out = []
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return out
    for n in names:
        p = f"{d}/{n}"
        if os.path.isdir(p) and re.match(r"^s\d+$", n) and not rel:
            out.append({"sample": n, "status": jread(f"{p}/status.json", {}), "files": files_of(p, n + "/")})
        elif os.path.isfile(p) and not n.endswith(".tmp") and ".tmp" not in n and n not in ("READY",) and not n.startswith("CLAIMED"):
            out.append({"name": rel + n, "size": os.path.getsize(p), "mtime": os.path.getmtime(p)})
    return out


AREFS = {}
AREF_F = os.environ.get("GENWEB_ACTOR_REFS", f"{LIVE}/posefix/score/out/actor_refs.json")
ALOE_REF = {"gestures_per_min": 10.9, "hands_moving_pct": 21.4, "cb0_repeat_pct": 58.3, "out_of_frame_pct": 21.1, "conf": 0.43}   # 10-27: + posefix REF aloe_heldout pose cols            # posefix run_stats.py REF aloe_heldout (24 real tracks, 10 min each)


def pose_refs(person):
    """10-26 user ("gestures ... first class metric"): {real: this actor's own GT clip (posefix actor_refs.json, read once per mtime), aloe: the
    held-out Aloe reference}; the clip is 12 s, so an actor's real gestures/min moves in steps of 5"""
    try:
        mt = os.path.getmtime(AREF_F)
        if AREFS.get("mt") != mt:
            with open(AREF_F) as f:
                AREFS.update(mt=mt, d=json.load(f))
    except (OSError, ValueError):
        pass
    a = (AREFS.get("d") or {}).get(person or "")
    lab = f"Person {str(person)[:2]}" if person and re.match(r"^\d\d_", str(person)) else (str(person or "actor"))
    real = {"label": lab + " real", **{k: a.get(k) for k in ("gestures_per_min", "hands_moving_pct", "cb0_repeat_pct", "out_of_frame_pct", "conf")}} if isinstance(a, dict) else None
    return {"real": real, "aloe": dict(ALOE_REF, label="Aloe")}


def render_total_s(d):
    """10-26: the render service's status carries no length: a run's full video = its frames x 80 ms (summary.json), else clip + prefill"""
    sm = jread(f"{d}/summary.json", {}) or {}
    if sm.get("frames"):
        return round(sm["frames"] * 0.08, 2)
    rq = jread(f"{d}/request.json", {}) or {}
    return round(float(rq.get("clip_s") or 0) + float((rq.get("prefill") or {}).get("s") or 0), 2) or None


VIDS_F, VIDX = f"{ROOT}/VIDS.json", {"mt": None, "d": {}, "r": {}}


def assign_vids():
    """10-28 user ("give each video generated an id so i can reference specific one"): every run gets a short stable id v<N>, a monotonic
    counter in creation order (run ids sort by time), kept in genweb/VIDS.json {rid: vid, "_next": N}; idempotent (new runs + backfill)"""
    import genids_api                                                   # 10-29 genids: ONE assigner (also run every 1 s by genids/assign.py);
    return genids_api.assign_vids(ROOT, RUNS)                           # agent / hidden runs get no v-id from 10-29; given ids never change


def vid_index():
    try:
        mt = os.path.getmtime(VIDS_F)
    except OSError:
        return VIDX
    if VIDX["mt"] != mt:
        d = jread(VIDS_F, {}) or {}
        VIDX.update(mt=mt, d={k: v for k, v in d.items() if not k.startswith("_")}, r={v: k for k, v in d.items() if not k.startswith("_")})
    return VIDX


def vid_of(rid):
    return vid_index()["d"].get(rid)


def live_weights():
    """10-28 (user: "set up backend server for both"): the extra monoplex weights whose worker is up: genweb/weights_live/<id>.json
    {id, label, short, worker, step, t} (written by the worker at load) -> {id: entry}"""
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/weights_live/*.json")):
        e = jread(f, {}) or {}
        if e.get("id"):
            out[str(e["id"])] = e
    return out


def _tag_of(m):
    m = str(m or "")
    if "duplex" in m.lower():
        step = re.search(r"(?:export-|step\s+)(\d+)", m)
        return "pose duplex" + (f" export-{step.group(1)}" if step else "")
    if "17850 SAM" in m or "mono17850sam" in m:
        return "monoplex 17850 SAM"
    if "17850 raw" in m or "mono17850" in m:
        return "monoplex 17850 raw"
    if "Xabi-data" in m or "aloe-v2pure" in m:
        return "32B Xabi-data FT · 17850 · export-200"
    if "export-357" in m:
        return "mono8020 export-357"
    if "Rosie" in m:
        return "Rosie FT · export-18"
    if "export-498" in m:
        return "pose duplex export-498"
    return m or None


def model_tag(d, req, st):
    """10-27 user ("where can i see which model this uses"): the model a run used = what its worker recorded (summary.json "model");
    a run not finished yet -> the model its weights are served by now (OPTIONS.json); None only when nothing is known"""
    t = _tag_of((jread(f"{d}/summary.json", {}) or {}).get("model"))
    if t:
        return t
    if str((st or {}).get("state") or "") in ("done", "error", "failed", "cancelled"):
        return None                                                     # finished but nothing recorded -> the page says "unknown"
    if (req.get("model") or "monoplex") == "duplex":
        return "pose duplex export-498"
    if (req.get("mono_weights") or "base") == "rosie":
        return "Rosie FT · export-18"
    lw = live_weights().get(str(req.get("mono_weights") or ""))
    if lw:
        return lw.get("short") or _tag_of(lw.get("label"))
    o = jread(f"{ROOT}/OPTIONS.json", {}) or {}
    return o.get("model_short") or _tag_of(o.get("model_label"))


def run_summary(rid, full=False):
    d = f"{RUNS}/{rid}"
    req = jread(f"{d}/request.json", {})
    st = jread(f"{d}/status.json", {})
    if not st:
        st = {"state": "queued" if os.path.exists(f"{d}/READY") else ("claimed" if any(n.startswith("CLAIMED") for n in os.listdir(d)) else "unknown")}
    o = {"run_id": rid, "created": req.get("created"), "state": st.get("state"), "progress": st.get("progress"), "headline": st.get("headline"),
         "msg": st.get("msg"), "error": st.get("error"), "params": req.get("params", {}), "voice": (req.get("prompt") or {}).get("voice"),
         "label": req.get("label"), "ui": req.get("ui") or {}, "seed": st.get("seed") or req.get("seed"), "ms_per_frame": st.get("ms_per_frame"),
         "model": req.get("model") or "monoplex", "batch": req.get("batch"), "stream": req.get("stream") is not False and not req.get("no_live"),
         "clip_s": req.get("clip_s"), "prefill": req.get("prefill") or {}, "mono_weights": req.get("mono_weights") or "base",
         "execution": st.get("execution")}
    o["model_tag"] = model_tag(d, req, st)
    o["ws"] = req.get("ws")
    o["vid"] = vid_of(rid)                                              # 10-28 user: short video id (v<N>)                                             # 10-28 user ("several instances of the page"): the run's workspace (None = default)                                # 10-27 user: which model every run used (card / batch / history)
    rs = rstatus(f"{ROUT}/{rid}.json")                                    # videort's render service status for this run
    rec = os.path.exists(f"{d}/avatar.mp4") and not os.path.exists(f"{ROUT}/{rid}.mp4")   # videort's live recording: approximate audio, stop-gap only
    o["render"] = {"state": "done" if rec else rs.get("state"), "progress_s": rs.get("progress_s"), "seconds": rs.get("seconds") or render_total_s(d),
                   "mp4": rec or os.path.exists(f"{ROUT}/{rid}.mp4"), "sxs": os.path.exists(f"{d}/avatar_sxs.mp4") or os.path.exists(f"{ROUT}/{rid}_sxs.mp4"),
                   "src": f"render/{rid}.flash.mp4" if os.path.exists(f"{ROUT}/{rid}.flash.mp4") else (f"runs/{rid}/avatar.mp4" if rec else (f"render/{rid}.mp4" if os.path.exists(f"{ROUT}/{rid}.mp4") else None)),
                   "flash": os.path.exists(f"{ROUT}/{rid}.flash.mp4"), "switch_s": switch_s(req),
                   "sxs_src": f"runs/{rid}/avatar_sxs.mp4" if os.path.exists(f"{d}/avatar_sxs.mp4") else (f"render/{rid}_sxs.mp4" if os.path.exists(f"{ROUT}/{rid}_sxs.mp4") else None)}
    if os.path.exists(f"{d}/stick.mp4"):                                 # 10-27 pose only: the stick figure IS the run's video (no avatar render)
        o["render"] = {"state": "done", "progress_s": None, "seconds": None, "mp4": True, "sxs": False, "src": f"runs/{rid}/stick.mp4",
                       "flash": False, "switch_s": None, "sxs_src": None, "stick": True}
    pf = (req.get("prefill") or {})
    if o["render"]["src"] and o.get("state") == "done" and o["render"].get("stick"):
        o["msg"] = ("pose only: stick figure (real speech, generated pose)" if (req.get("params") or {}).get("pose_only")
                    else "stick figure video (this actor has no avatar)")   # 10-27: Rosie runs render as a stick figure too
    elif o["render"]["src"] and o.get("state") == "done":
        o["msg"] = ("live recording shown (its audio is the live stream's approximate decode); the exact replay is rendering"
                    if rec else "avatar video ready")
        o["render"]["approx"] = bool(rec)
    elif o.get("state") == "done":
        r_ = o["render"]
        o["msg"] = ("avatar render failed: " + str(rs.get("error") or "")[:200]) if r_.get("state") == "fail" else (
            f"making the avatar video · {r_.get('progress_s') or 0:.0f} / {r_.get('seconds') or 0:.0f} s" if r_.get("state") == "rendering" else "avatar render queued")
    o["person"] = pf.get("person") or req.get("person")
    o["hidden"] = bool(req.get("hidden"))                                  # 10-26: agent test run (never listed)
    gp = o["person"]                                                    # 10-26: the GT clip, preferring its sequential-sampler re-render
    gs = f"render_seq/gt-{gp}.mp4" if gp and os.path.exists(f"{RSEQ}/gt-{gp}.mp4") else (f"render/gt-{gp}.mp4" if gp and os.path.exists(f"{ROUT}/gt-{gp}.mp4") else None)
    o["gt_mp4"], o["gt_src"] = bool(gs), gs
    o["pose"] = history_api.pose_stats(d)                                 # 10-26 user: cb0 repeat % over the generated frames
    if o["pose"] is not None:
        o["pose"]["ref"] = pose_refs(o.get("person"))
        if o["pose"].get("timing_kappa_real") is not None:              # 10-27: gesturemetric's REAL-Aloe-clips kappa ref (not per actor)
            o["pose"]["ref"]["aloe"]["timing_kappa"] = o["pose"]["timing_kappa_real"]                    # 10-26 user: gestures/min + hands moving % next to real refs
    et = history_api.eta_text(rid, o)                                    # 10-26 user: honest ETAs (queue position + start / render done times)
    if et:
        o["msg"] = et
        if o.get("state") in (None, "queued", "unknown"):
            st = dict(st, msg=et)
    o["summary"] = jread(f"{d}/summary.json") if full else None
    if full:
        o.update(status=st, request=req, files=files_of(d))
    return o


def prefill_data():
    """10-27 picker (user: "pick the person ... + raw codes / processed codes"): per actor, which prefill data is on disk, by the worker's own
    file rules: raw = openings/<id> (pose + mimi; else the worker uses the actor clip), sam = openings-sam/<id> (pose + mimi + meta),
    sam_orig = its pre re-cut window; *_video = the opening's SUMMARY video flag (real-video avatar prefill); thumb = render/thumb-<id>.jpg"""
    out = {}
    for root, k in (("openings", "raw"), ("openings-sam", "sam")):
        summ = jread(f"{ROOT}/{root}/SUMMARY.json", {}) or {}
        need = ("pose_tokens.safetensors", "mimi_local.safetensors") + (("meta.json",) if k == "sam" else ())
        try:
            ids = [d for d in os.listdir(f"{ROOT}/{root}") if os.path.isdir(f"{ROOT}/{root}/{d}") and not d.startswith(("_", "."))]
        except OSError:
            ids = []
        for d in ids:
            if all(os.path.isfile(f"{ROOT}/{root}/{d}/{x}") for x in need):
                e = out.setdefault(d, {})
                e[k], e[f"{k}_video"] = True, bool((summ.get(d) or {}).get("video"))
                if k == "sam" and os.path.isfile(f"{ROOT}/{root}/{d}/orig/pose_tokens.safetensors"):
                    e["sam_orig"] = True
    for e in out.values():
        e.setdefault("raw", False)
        e.setdefault("sam", False)
    try:
        for f in os.listdir(ROUT):
            m = re.match(r"^thumb-([0-9A-Za-z_]+)\.jpg$", f)
            if m:
                out.setdefault(m.group(1), {"raw": False, "sam": False})["thumb"] = True
    except OSError:
        pass
    return out


async def index(req):
    return web.FileResponse(f"{HERE}/index.html", headers={"Cache-Control": "no-store"})


_DUPOK = {"v": False}


def _dup_ready():
    """10-28 (relaunch with monoplex only; the bare DUP_ENABLED flag said duplex was ready with no worker, so picking it hung): duplex is
    ready only while DUP_ENABLED exists AND one of our RUNNING jobs has a duplex worker log (logs/worker-dup-<job>.log, written by
    unit.sbatch dup / dup_hold.sbatch / dupui.sbatch). squeue failure -> the last known answer"""
    if not os.path.exists(f"{ROOT}/DUP_ENABLED"):
        _DUPOK["v"] = False
        return False
    if os.environ.get("GENWEB_RUNTIME") == "systemd":
        return worker_ready(f"{ROOT}/duplex_ready.json")
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-t", "RUNNING", "-o", "%i"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            _DUPOK["v"] = any(os.path.exists(f"{ROOT}/logs/worker-dup-{j}.log") for j in out.stdout.split())
    except (OSError, subprocess.SubprocessError):
        pass
    return _DUPOK["v"]


def worker_ready(path):
    """Only advertise a VM worker after load, with a live process and fresh heartbeat."""
    try:
        info = json.load(open(path))
        stamp = float(info.get("time", info.get("timestamp", os.path.getmtime(path))))
        if time.time() - stamp > 30 or info.get("ready") is False:
            return False
        os.kill(int(info["pid"]), 0)
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


async def health(req):
    mono_path = f"{ROOT}/mono_ready.json"
    mono, duplex = worker_ready(mono_path), _dup_ready()
    return web.json_response({"web": "ok", "monoplex_ready": mono, "duplex_ready": duplex,
                              "monoplex": jread(mono_path, {}), "duplex": jread(f"{ROOT}/duplex_ready.json", {})},
                             headers={"Cache-Control": "no-store"})


async def schema(req):
    sc = jread(f"{ROOT}/SCHEMA.json")
    scd = jread(f"{ROOT}/DUPLEX_SCHEMA.json")                           # 10-25: the duplex worker's knobs (model picker), once it exists
    opts = jread(f"{ROOT}/OPTIONS.json", {})
    mono_cfg = jread(f"{ROOT}/MONO_CK.json", {})
    opts["model_prompt"] = mono_cfg.get("prompt", "mono8020")
    opts["supports_persona"] = opts["model_prompt"] != "v2pure"
    def _busy():                                                        # 10-26: off the event loop (a slow /data scan froze every request)
        n = 0
        try:
            for rid in os.listdir(RUNS):
                if jread(f"{RUNS}/{rid}/request.json", {}).get("hidden"):
                    continue
                s = jread(f"{RUNS}/{rid}/status.json", {})
                if s.get("state") in ("queued", "running") or os.path.exists(f"{RUNS}/{rid}/READY"):
                    n += 1
        except OSError:
            pass
        return n
    busy = await offload(("busy",), 2.0, _busy)
    try:
        av = open(f"{LIVE}/AVATAR_URL.txt").read().split()[0]
    except (OSError, IndexError):
        av = ""
    op = {}                                                             # 10-25: per actor, the conversation-start prefill on disk
    for k, v in (jread(f"{ROOT}/openings/SUMMARY.json", {}) or {}).items():
        if isinstance(v, dict):
            op[k] = {"legit": bool(v.get("legit_opening")) and os.path.isfile(f"{ROOT}/openings/{k}/pose_tokens.safetensors"),
                     "t0": v.get("t0"), "T": v.get("T"), "video": bool(v.get("video")), "words": v.get("n_words")}
    pfd = await offload(("pfdata",), 30.0, prefill_data)
    dup_ok = await offload(("dupready",), 15.0, _dup_ready)
    return web.json_response({"schema": sc, "schema_ok": isinstance(sc, list) and len(sc) > 0, "options": opts, "weights": list(live_weights().values()), "pending": busy, "avatar_url": av, "openings": op, "prefill_data": pfd, "prefill_codes_ok": os.path.exists(f"{ROOT}/PREFILL_CODES_OK"), "schema_dup": scd if isinstance(scd, list) and scd else None,
                              "dup_ready": dup_ok, "renderers": RENDERERS, "batch_stream": BATCH_STREAM,
                              "vansh_live": os.path.exists(VANSH_LIVE),
                              "vansh_note": (open(f"{ROOT}/VANSH_NOTE.txt").read().strip()[:120] if os.path.exists(f"{ROOT}/VANSH_NOTE.txt") else "renders only"),
                              "schema_mtime": os.path.getmtime(f"{ROOT}/SCHEMA.json") if os.path.exists(f"{ROOT}/SCHEMA.json") else None},
                             headers={"Cache-Control": "no-store"})


async def gen(req):
    d = await req.json()
    renderer = str(d.get("video_weights") or "param")
    if renderer not in RENDERERS and not (renderer == "both" and {"param", "vansh"}.issubset(RENDERERS)):
        return web.json_response({"error": "That video renderer is not hosted here. Select Param."}, status=400)
    if not BATCH_STREAM and d.get("stream") in (True, 1, "1", "true"):
        return web.json_response({"error": "Batch video is rendered after generation. Live mic remains available in Duplex."}, status=400)
    params = d.get("params") if isinstance(d.get("params"), dict) else {}
    prompt = dict(d.get("prompt") or {})
    v = str(prompt.get("voice") or "")
    if v.startswith("upload:"):                                         # the page's upload id -> the absolute wav path for the worker
        h = v[7:]
        if not re.match(r"^[0-9a-f]{12}$", h) or not os.path.isfile(f"{UPL}/{h}.wav"):
            return web.json_response({"error": "uploaded clip not found: upload it again"}, status=400)
        prompt["voice"] = f"upload:{UPL}/{h}.wav"
    voice = prompt.get("voice")
    if d.get("model") == "duplex" and not await offload(("dupready",), 15.0, _dup_ready):   # 10-28: no duplex worker -> never queue a run that hangs
        return web.json_response({"error": "Duplex is not hosted right now (no duplex worker running)."}, status=409)
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    rd = f"{RUNS}/{rid}"
    os.makedirs(rd)
    model = "duplex" if d.get("model") == "duplex" else "monoplex"
    extra = {k: d[k] for k in ("conv", "side", "a_after", "person") if model == "duplex" and k in d}
    if model == "monoplex" and str(d.get("mono_weights") or "base") == "rosie":   # 10-27 user ("put up rosie"): the Rosie FT weights' worker
        extra["mono_weights"] = "rosie"
    elif model == "monoplex" and str(d.get("mono_weights") or "base") in live_weights():   # 10-28: any announced monoplex weights id
        extra["mono_weights"] = str(d.get("mono_weights"))
    # 10-26 user: "Stream live" off -> the worker never opens videort's live feed (its existing no_live switch: no start / f / end, no
    # recording); generation and the exact render are unchanged. Batch = the Generate press this run belongs to.
    # 10-26 user ("also dont livestream generations as i said"): OFF unless the request explicitly asks for it (page: the user switched it on)
    if d.get("stream") in (True, 1, "1", "true"):
        extra.update(stream=True, stream_set_by_user=True)
    else:
        extra.update(stream=False, no_live=True)
    if d.get("gt") in (True, 1, "1", "true"):                          # 10-26 user: GT video only when the user shows it (on-demand GT render)
        extra["gt"] = True
    # 10-26 user: VIDEO WEIGHTS (default Param student; Vansh = his checkpoint through the same sequential sampler at his settings, rendered
    # by videort's Vansh renderers from render/jobs_vansh). Vansh runs never stream until videort's live Vansh backend exists (VANSH_LIVE).
    vw = str(d.get("video_weights") or "").lower()
    extra["video_weights"] = vw if vw in ("vansh", "both") else "param"
    if extra["video_weights"] != "param":                               # 10-26 user: Vansh at 20 steps by default, 50 on request
        extra["vansh_steps"] = 50 if str(d.get("vansh_steps")) == "50" else 20
    # 10-26 user: "both" = generate ONCE, render with Param (render/jobs -> <rid>.mp4) AND Vansh (jobs_vansh -> <rid>.vansh.mp4, queued by
    # the compare path as soon as the feed exists), shown side by side; never streamed
    if (extra["video_weights"] == "vansh" and not os.path.exists(VANSH_LIVE)) or extra["video_weights"] == "both":
        extra.update(stream=False, no_live=True)
        extra.pop("stream_set_by_user", None)
    b = d.get("batch") if isinstance(d.get("batch"), dict) else {}
    if BID.match(str(b.get("id") or "")):
        try:
            extra["batch"] = {"id": str(b["id"]), "n": max(1, min(64, int(b.get("n") or 1))), "k": max(0, int(b.get("k") or 0))}
        except (TypeError, ValueError):
            pass
    jwrite(f"{rd}/request.json", {"run_id": rid, "created": time.time(), "model": model, **extra, "seed": int(d.get("seed") or 0),
                                  "clip_s": float(d.get("clip_s") or 20), "prefill": d.get("prefill") or {}, "prompt": prompt, "params": params,
                                  "ui": d.get("ui") or {}, "label": str(d.get("label") or "")[:120], "client": "genweb",
                                  **({"hidden": True} if d.get("hidden") is True else {}),
                                  **({"ws": str(d["ws"])[:40]} if re.match(r"^[A-Za-z0-9_-]{1,40}$", str(d.get("ws") or "")) else {}),   # 10-28 user: workspaces
                                  **({"prefill_codes": d["prefill_codes"]} if model == "monoplex" and d.get("prefill_codes") in ("raw", "sam") else {}),   # 10-27 picker: absent = the model's own codes
                                  **({"debug_inputs": d["debug_inputs"]} if d.get("hidden") is True and isinstance(d.get("debug_inputs"), dict) else {})})   # 10-28: hidden 2x2 isolation only   # 10-27 fix: agent test runs posted here stay hidden
    jwrite(f"{rd}/status.json", {"state": "queued", "progress": 0, "msg": "waiting for a free worker"})
    if extra["video_weights"] == "both":
        jwrite(f"{rd}/compare.json", {"other": "vansh", "at": time.time(), "why": "video weights: both"})
    try:
        assign_vids()                                                   # 10-28: the new run's short id, before it is queued
    except Exception as e:  # noqa: BLE001
        print(f"[genweb] vid assign failed: {e}", flush=True)
    open(f"{rd}/READY", "w").close()
    print(f"[genweb {time.strftime('%H:%M:%S')}] run {rid} queued: voice {voice} params {json.dumps(params)[:300]}", flush=True)
    return web.json_response({"run_id": rid})


async def upload(req):
    data = await req.read()
    if not data or len(data) > MAX_UPLOAD:
        return web.json_response({"error": f"upload must be 1 B - {MAX_UPLOAD >> 20} MB"}, status=400)
    h = secrets.token_hex(6)
    with open(f"{UPL}/{h}.wav", "wb") as f:
        f.write(data)
    return web.json_response({"id": h, "bytes": len(data)})


_OC, _OF = {}, {}


async def offload(key, ttl, fn):
    """10-26 (the page hung: every open page polls these scans every 1.5-2 s, and they ran ON the event loop, so one slow scan froze every
    request for 20 s): run the scan in a worker thread, share one result between all callers within ttl seconds, coalesce concurrent misses"""
    hit = _OC.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    fut = _OF.get(key)
    if fut is None:
        fut = _OF[key] = asyncio.ensure_future(asyncio.to_thread(fn))
        fut.add_done_callback(lambda f, k=key: (_OF.pop(k, None), f.exception() is None and _OC.__setitem__(k, (time.time(), f.result()))))
    return await asyncio.shield(fut)


async def runs(req):
    n = int(req.query.get("n", 200))
    def scan():
        try:
            ids = sorted((r for r in os.listdir(RUNS) if RID.match(r) and not is_hidden(r)), reverse=True)[:n]   # 10-26: never agent runs
        except OSError:
            ids = []
        return {"runs": [run_summary(r) for r in ids]}
    return web.json_response(await offload(("runs", n), 1.0, scan), headers={"Cache-Control": "no-store"})


async def vid_lookup(req):
    v = req.match_info["vid"].strip().lower()
    rid = vid_index()["r"].get(v)
    if not rid or not os.path.isdir(f"{RUNS}/{rid}"):
        return web.json_response({"error": "no such video id"}, status=404)
    q = jread(f"{RUNS}/{rid}/request.json", {}) or {}
    return web.json_response({"vid": v, "run_id": rid, "ws": q.get("ws"), "model": q.get("model") or "monoplex", "mono_weights": q.get("mono_weights") or "base",
                              "batch": (q.get("batch") or {}).get("id"), "hidden": bool(q.get("hidden"))})


async def run(req):
    rid = req.match_info["rid"]
    if re.match(r"^v[0-9]{1,7}$", rid):                                 # 10-28: look a run up by its short id
        rid = vid_index()["r"].get(rid) or rid
    if not RID.match(rid) or not os.path.isdir(f"{RUNS}/{rid}"):
        return web.json_response({"error": "no such run"}, status=404)
    o = run_summary(rid, full=True)
    o["video_weights"] = (o.get("request") or {}).get("video_weights") or "param"             # 10-26: the main player's Param | Vansh pair
    o["cmp"] = cmp_info(rid, "vansh" if o["video_weights"] == "vansh" else "param", (o.get("render") or {}).get("src"))
    o["vansh_steps"] = vsteps(rid, o.get("request") or {}) if o["video_weights"] != "param" or os.path.exists(f"{ROUT}/{rid}.vansh.mp4") or os.path.exists(f"{RUNS}/{rid}/vansh_steps.json") else None
    o["vprog"] = vansh_eta(rid, d, rs, vdir(vsteps(rid, o.get("request") or {}))) if o["video_weights"] == "vansh" and not (o.get("render") or {}).get("src") else None   # 10-26 Vansh-only ETA
    return web.json_response(o, headers={"Cache-Control": "no-store"})


# 10-26 user (via main): BATCH view + full HISTORY, both built from the run dirs on disk on every request (so they survive restarts).
# A card plays the EXACT render only (the flash + beep version once burnt, else the render service's mp4), never the live recording.
def exact_src(rid):
    for n in (f"{rid}.flash.mp4", f"{rid}.mp4"):
        if os.path.exists(f"{ROUT}/{n}"):
            return f"render/{n}", f"{ROUT}/{n}"
    return None, None


def card_state(o, src):
    """queued / generating / rendering / done / failed + a short note"""
    st, r = o.get("state"), o.get("render") or {}
    et = None if src else history_api.eta_text(o.get("run_id"), o)          # 10-26 user: honest ETAs
    if st == "error":
        e = str(o.get("error") or "failed")
        return "failed", ("cancelled" if "cancel" in e.lower() else e[:200])   # 10-27 user: say "cancelled", not RuntimeError: cancelled
    if st == "done":
        if src:
            return "done", ""
        if r.get("state") == "fail":
            return "failed", "avatar render failed"
        if r.get("state") == "rendering":
            return "rendering", et or f"making the avatar video · {r.get('progress_s') or 0:.0f} / {r.get('seconds') or 0:.0f} s"
        return "rendering", et or "render queued"
    if st == "live":                                                    # 10-26 dupui livedup: a live call in progress
        return "generating", "live call in progress"
    if st == "importing":
        return "generating", "saving the call recording"
    if st == "running":
        return "generating", f"generating {int(100 * (o.get('progress') or 0))}%"
    if st == "claimed":
        return "generating", "starting"
    return "queued", et or "waiting for a free worker"


_HID = {}


def is_hidden(rid):
    """10-26 user: agent test runs carry request.json "hidden": true; no listing ever shows them (cached by the request's mtime)"""
    p = f"{RUNS}/{rid}/request.json"
    try:
        m = os.path.getmtime(p)
    except OSError:
        return False
    h = _HID.get(rid)
    if h and h[0] == m:
        return h[1]
    v = bool((jread(p, {}) or {}).get("hidden"))
    _HID[rid] = (m, v)
    return v


def run_ids():
    try:
        return sorted((r for r in os.listdir(RUNS) if RID.match(r) and not is_hidden(r)), reverse=True)
    except OSError:
        return []


def group_of(o):
    b = o.get("batch") or {}
    return str(b["id"]) if isinstance(b, dict) and BID.match(str(b.get("id") or "")) else o["run_id"]


def card(o, words=False):
    src, _ = exact_src(o["run_id"])
    state, note = card_state(o, src)
    c = {"run_id": o["run_id"], "k": (o.get("batch") or {}).get("k", 0), "seed": o.get("seed"), "state": state, "note": note,
         "progress": o.get("progress"), "src": src, "headline": o.get("headline"), "created": o.get("created"), "execution": o.get("execution"),
         "rp": [(o.get("render") or {}).get("progress_s"), (o.get("render") or {}).get("seconds")]}
    c["pose"] = o.get("pose")                                             # 10-26 user: cb0 repeat % (generated frames)
    c["video_weights"] = (jread(f"{RUNS}/{o['run_id']}/request.json", {}) or {}).get("video_weights") or "param"
    c["cmp"] = cmp_info(o["run_id"], "vansh" if c["video_weights"] == "vansh" else "param", src)   # "both": Param main + Vansh other
    _rq = jread(f"{RUNS}/{o['run_id']}/request.json", {}) or {}
    c["vansh_steps"] = vsteps(o["run_id"], _rq) if c["video_weights"] != "param" or c["cmp"].get("vansh") or c["cmp"].get("other_state") else None   # 10-26: "Vansh · 20 steps"
    c["vprog"] = o.get("vprog")
    if words:
        ws = jread(f"{RUNS}/{o['run_id']}/words.json", []) or []
        c["pf_text"] = " ".join(str(w.get("w", "")) for w in ws if isinstance(w, dict) and w.get("prefill"))[:600]
        c["text"] = " ".join(str(w.get("w", "")) for w in ws if isinstance(w, dict) and not w.get("prefill") and w.get("ch", "B") == "B")[:1200]   # 10-26 dupui: B only
        c["text_a"] = " ".join(str(w.get("w", "")) for w in ws if isinstance(w, dict) and not w.get("prefill") and w.get("ch") == "A")[:1200]   # duplex: A's words apart
        c["request"] = jread(f"{RUNS}/{o['run_id']}/request.json", {})
    c["model_tag"] = o.get("model_tag")                                 # 10-27 user: which model made this run
    return c


VPROG = {}                               # 10-26: first sighting (time, progress_s) of each Vansh render, for the ETA


def vansh_eta(name, d, rs, jobdir):
    """10-26 user (Vansh renders are slow): {state, progress_s, seconds, eta_s} while rendering (rate observed since first seen; under 5 s of
    observed progress assume 4x slower than real time), {state: queued, ahead: n} while its job waits (older unfinished jobs in the queue)"""
    secs = rs.get("seconds") or render_total_s(d)
    vs = 20 if jobdir == RJV else 50                                   # 10-26: Vansh steps of this queue (the label says it)
    if rs.get("state") == "rendering":
        p, now = float(rs.get("progress_s") or 0), time.time()
        t0, p0 = VPROG.setdefault(name, (now, p))
        rate = (p - p0) / (now - t0) if p - p0 >= 5 and now > t0 else (0.8 if vs == 20 else 0.25)   # 20 steps ~1.25x slower than real time
        return {"state": "rendering", "progress_s": round(p, 1), "seconds": secs, "steps": vs, "eta_s": round(max(0.0, (secs or 0) - p) / max(rate, 1e-3))}
    fp = f"{jobdir}/{name}.feed.jsonl"
    if os.path.exists(fp) and not os.path.exists(f"{ROUT}/{name}.mp4"):
        try:
            mt = os.path.getmtime(fp)
            ahead = sum(1 for x in os.listdir(jobdir) if x.endswith(".feed.jsonl") and x != f"{name}.feed.jsonl"
                        and os.path.getmtime(f"{jobdir}/{x}") < mt and not os.path.exists(f"{ROUT}/{x[:-11]}.mp4"))
        except OSError:
            ahead = None
        return {"state": "queued", "ahead": ahead, "seconds": secs, "steps": vs}
    return {"state": rs.get("state"), "seconds": secs} if rs.get("state") else None


def cmp_info(rid, vw, src):
    """10-26 user (compare weights): the same feed re-rendered with the other video weights -> render/out/<rid>.<other>.mp4"""
    other = "param" if vw == "vansh" else "vansh"
    n = f"{rid}.{other}"
    osrc = f"render/{n}.mp4" if os.path.exists(f"{ROUT}/{n}.mp4") else None
    st = "done" if osrc else (rstatus(f"{ROUT}/{n}.json") or {}).get("state")
    if not st and (any(os.path.exists(f"{d}/{n}.feed.jsonl") for d in ((RJV, RJV50) if other == 'vansh' else (ROOT + '/render/jobs',))) or os.path.exists(f"{RUNS}/{rid}/compare.json")):
        st = "queued"
    prog = vansh_eta(n, f"{RUNS}/{rid}", rstatus(f"{ROUT}/{n}.json") or {}, vdir(vsteps(rid, jread(f"{RUNS}/{rid}/request.json", {}) or {}))) if other == "vansh" and not osrc else None
    return {vw: src, other: osrc, "other": other, "other_state": st, "other_prog": prog}


def queue_compare(rid, steps=None):
    """queue the run's own feed for a re-render with the other weights; 'queued' / 'exists' / 'pending' (run not finished) / 'failed'
    10-26: a Vansh re-render uses the request's vansh_steps (Both), else the compare's steps (default 20), recorded in vansh_steps.json"""
    rq = jread(f"{RUNS}/{rid}/request.json", {}) or {}
    other = "param" if rq.get("video_weights") == "vansh" else "vansh"
    if other == "vansh" and rq.get("vansh_steps") not in (20, 50):
        cj = jread(f"{RUNS}/{rid}/compare.json", {}) or {}
        steps = steps if steps in (20, 50) else (cj.get("steps") if cj.get("steps") in (20, 50) else (jread(f"{RUNS}/{rid}/vansh_steps.json", {}) or {}).get("steps"))
        steps = steps if steps in (20, 50) else 20
    else:
        steps = rq.get("vansh_steps") if rq.get("vansh_steps") in (20, 50) else 50
    n, jd = f"{rid}.{other}", (vdir(steps) if other == "vansh" else f"{ROOT}/render/jobs")
    if os.path.exists(f"{ROUT}/{n}.mp4") or any(os.path.exists(f"{d}/{n}.feed.jsonl") for d in ((RJV, RJV50) if other == "vansh" else (jd,))) or os.path.exists(f"{ROUT}/{n}.claim"):
        return "exists"
    st, fp = (jread(f"{RUNS}/{rid}/status.json", {}) or {}).get("state"), f"{RUNS}/{rid}/feed.jsonl"
    if st == "error":
        return "failed"
    if st != "done" or not os.path.exists(fp):
        jwrite(f"{RUNS}/{rid}/compare.json", {"other": other, "at": time.time(), **({"steps": steps} if other == "vansh" else {})})
        return "pending"
    if other == "vansh" and rq.get("vansh_steps") not in (20, 50):
        jwrite(f"{RUNS}/{rid}/vansh_steps.json", {"steps": steps, "at": time.time()})
    os.makedirs(jd, exist_ok=True)
    with open(fp, "rb") as fi, open(f"{jd}/{n}.feed.jsonl.tmp", "wb") as fo:
        fo.write(fi.read())
    os.replace(f"{jd}/{n}.feed.jsonl.tmp", f"{jd}/{n}.feed.jsonl")
    return "queued"


def comparer():
    """runs asked to compare before they finished: queue their other-weights re-render once the feed exists"""
    while True:
        try:
            for m in glob.glob(f"{RUNS}/*/compare.json"):
                rid = os.path.basename(os.path.dirname(m))
                if queue_compare(rid) in ("queued", "exists", "failed"):
                    os.remove(m)
        except Exception as e:  # noqa: BLE001
            print(f"[genweb] comparer: {e}", flush=True)
        time.sleep(2.0)


async def compare(req):
    if not {"param", "vansh"}.issubset(RENDERERS):
        return web.json_response({"error": "Video-weight comparison requires both renderers; only Param is hosted here."}, status=409)
    """POST /api/compare/<batch or run id>: every run of it also rendered with the other video weights (same feed, side by side)"""
    cards = batch_cards(req.match_info["bid"])
    if not cards:
        return web.json_response({"error": "no such batch"}, status=404)
    try:
        q = await req.json() if req.can_read_body else {}
    except Exception:  # noqa: BLE001
        q = {}
    st = 50 if str((q or {}).get("vansh_steps") or req.query.get("vansh_steps") or "") == "50" else 20   # 10-26: compare at 20 steps unless 50 asked
    res = {c["run_id"]: queue_compare(c["run_id"], st) for c in cards}
    return web.json_response({"n": len(res), "res": res})


# GET /api/history: history_api.py (10-26 History tab; supersedes the first version that lived here)

def batch_cards(bid):
    if RID.match(bid):
        ids = [bid] if os.path.isdir(f"{RUNS}/{bid}") and not is_hidden(bid) else []
    elif BID.match(bid):
        ids = [rid for rid in run_ids() if ((jread(f"{RUNS}/{rid}/request.json", {}) or {}).get("batch") or {}).get("id") == bid]
    elif bid[:1] == "L" and RID.match(bid[1:]):                         # 10-26: an older n_samples press (before batch ids), as History groups it
        import history_api
        ids = history_api.legacy_runs(bid)
    else:
        ids = []
    cards = [card(run_summary(rid), words=True) for rid in ids]
    cards.sort(key=lambda c: (c["k"] or 0, c["run_id"]))
    return cards


async def batch(req):
    bid = req.match_info["bid"]
    cards = await offload(("batch", bid), 1.0, lambda: batch_cards(bid))   # 10-26: off the event loop
    if not cards:
        return web.json_response({"error": "no such batch"}, status=404)
    st = [c["state"] for c in cards]
    return web.json_response({"id": bid, "batch": bool(BID.match(bid)) or bid[:1] == "L", "n": len(cards), "done": st.count("done"), "failed": st.count("failed"),
                              "created": min((c["created"] for c in cards if c["created"]), default=None), "cards": cards},
                             headers={"Cache-Control": "no-store"})


async def batch_zip(req):
    bid = req.match_info["bid"]
    if not (BID.match(bid) or re.match(r"^L[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$", bid)):
        return web.json_response({"error": "bad batch id"}, status=400)
    path = await asyncio.to_thread(_batch_zip_file, bid)                # 10-26: built in a thread, on disk (never in RAM, never on the event loop)
    return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="{bid}.zip"', "Cache-Control": "no-store"})


def _batch_zip_file(bid):
    cards = batch_cards(bid)
    os.makedirs(f"{ROOT}/render/zips", exist_ok=True)
    path = f"{ROOT}/render/zips/{bid}.zip"
    with zipfile.ZipFile(path + ".tmp", "w", zipfile.ZIP_STORED) as z:
        for i, c in enumerate(cards):
            _, p = exact_src(c["run_id"])
            if p:
                z.write(p, arcname=f"{bid}_{i + 1:02d}_seed{c['seed']}_{c['run_id']}{'_vansh' if c.get('video_weights') == 'vansh' else ''}.mp4")
            o = (c.get("cmp") or {}).get("other")                       # 10-26: the compare re-render with the other weights, if any
            if o and os.path.exists(f"{ROUT}/{c['run_id']}.{o}.mp4"):
                z.write(f"{ROUT}/{c['run_id']}.{o}.mp4", arcname=f"{bid}_{i + 1:02d}_seed{c['seed']}_{c['run_id']}_{o}.mp4")
            for n in ("mix.wav", "a.wav", "b.wav", "a_own.wav"):          # 10-26 dupui: a duplex run's conversation audio too
                f = f"{RUNS}/{c['run_id']}/{n}"
                if (c.get("request") or {}).get("model") == "duplex" and os.path.isfile(f):
                    z.write(f, arcname=f"{bid}_{i + 1:02d}_seed{c['seed']}_{c['run_id']}_{n}")
    os.replace(path + ".tmp", path)
    return path


async def changelog(req):
    d = jread(f"{ROOT}/CHANGELOG.json", {}) or {}                      # 10-26 user: what version is running (read from disk on every request)
    d["live"] = {"server": SERVER_V, **LOADED}
    return web.json_response(d, headers={"Cache-Control": "no-store"})


async def cancel(req):
    rid = req.match_info["rid"]
    rd = f"{RUNS}/{rid}"
    if not RID.match(rid) or not os.path.isdir(rd):
        return web.json_response({"error": "no such run"}, status=404)
    open(f"{rd}/CANCEL", "w").close()                                   # the worker checks for it; an unclaimed run is just dropped
    if os.path.exists(f"{rd}/READY"):
        os.remove(f"{rd}/READY")
        jwrite(f"{rd}/status.json", {"state": "error", "error": "cancelled before it started"})
    rc = False                                                          # 10-27 user ("why cant i cancel it"): a generated run whose avatar video
    st = jread(f"{rd}/status.json", {})                                 # is still QUEUED for rendering: drop its render job (a render that has
    if st.get("state") == "done" and not os.path.exists(f"{ROUT}/{rid}.claim"):   # started is left to finish)
        for jd in (f"{ROOT}/render/jobs", RJV, RJV50):
            j = f"{jd}/{rid}.feed.jsonl"
            if os.path.isfile(j):
                os.remove(j)
                rc = True
        if rc:
            jwrite(f"{rd}/status.json", dict(st, state="error", error="cancelled before its video rendered"))
    return web.json_response({"ok": True, "render_cancelled": rc})


async def renderseqfile(req):
    path = req.match_info["path"]                                       # 10-26: GT clips' sequential-sampler re-renders only
    if not re.match(r"^gt-[0-9a-z_]+\.mp4$", path) or not os.path.isfile(f"{RSEQ}/{path}"):
        raise web.HTTPNotFound()
    return web.FileResponse(f"{RSEQ}/{path}", headers={"Cache-Control": "no-store"})


async def renderfile(req):
    path = req.match_info["path"]
    if ".." in path or "/" in path or not (path.endswith(".mp4") or re.match(r"^thumb-[0-9A-Za-z_]+\.jpg$", path)):   # 10-27 picker: actor reel thumbs
        raise web.HTTPNotFound()
    p = f"{ROUT}/{path}"
    if not os.path.isfile(p):
        raise web.HTTPNotFound()
    return web.FileResponse(p, headers={"Cache-Control": "no-store"})


async def runfile(req):
    rid, path = req.match_info["rid"], req.match_info["path"]
    if not RID.match(rid) or ".." in path or path.startswith("/"):
        raise web.HTTPNotFound()
    p = f"{RUNS}/{rid}/{path}"
    if not os.path.isfile(p):
        raise web.HTTPNotFound()
    return web.FileResponse(p, headers={"Cache-Control": "no-store"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9205)
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(UPL, exist_ok=True)
    app = web.Application(client_max_size=MAX_UPLOAD + (1 << 20))
    app.router.add_get("/", index)
    app.router.add_get("/index.html", index)
    app.router.add_get("/health", health)
    app.router.add_get("/api/schema", schema)
    app.router.add_post("/api/gen", gen)
    app.router.add_post("/api/upload", upload)
    app.router.add_get("/api/runs", runs)
    app.router.add_get("/api/run/{rid}", run)
    import pose_overlay_api; pose_overlay_api.setup(app, globals())
    import audit_api; audit_api.setup(app)
    app.router.add_get("/api/vid/{vid}", vid_lookup)                    # 10-28: v-ID -> run id (+ ws, model tag) for ?v= links and lookups
    app.router.add_get("/api/batch/{bid}", batch)
    app.router.add_get("/api/batch/{bid}/zip", batch_zip)
    app.router.add_post("/api/compare/{bid}", compare)
    app.router.add_get("/api/changelog", changelog)
    app.router.add_post("/api/cancel/{rid}", cancel)
    app.router.add_get("/runs/{rid}/{path:.+}", runfile)
    app.router.add_get("/render/{path}", renderfile)
    app.router.add_get("/render_seq/{path}", renderseqfile)
    import history_api; history_api.setup(app, globals())             # 10-26 user: History tab (/api/history, /api/thumb/<rid>.jpg)
    import dupui_api; dupui_api.setup(app, globals())                 # 10-26 dupui: /api/dupzip/<bid> (duplex batch zip + conversation audio)
    import livedup_api; livedup_api.setup(app, globals())             # 10-26 dupui livedup: live duplex calls -> runs / History (/api/livedup/*)
    import voiceclip_api; voiceclip_api.setup(app, globals())         # 10-28 user: hear the voice clip (/api/voiceclip, /api/voiceclip/run/<rid>, wav/)
    import genids_api; genids_api.setup(app, globals())               # 10-29 user: G<n> / B<n> ids (/api/genids, /api/gid/<G142|B37>; GEN_IDS.json)
    import prefs_api; prefs_api.setup(app, globals())                 # 10-29 user: last-used settings on the server (/api/prefs; state/prefs.json)
    threading.Thread(target=flasher, daemon=True).start()
    threading.Thread(target=reaper, daemon=True).start()               # 10-26: re-queue runs whose worker died
    threading.Thread(target=comparer, daemon=True).start()
    os.makedirs(RJV, exist_ok=True)
    os.makedirs(RJV50, exist_ok=True)
    print(f"[genweb] serving {ROOT} on :{a.port}", flush=True)
    web.run_app(app, host="127.0.0.1", port=a.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
