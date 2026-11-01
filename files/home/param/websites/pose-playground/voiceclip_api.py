"""10-28 user ("how do i know what the clip you are using for cloning is i want to be able to hear it"): the monoplex voice clip, audible.
Registered by server.py: voiceclip_api.setup(app, globals()). The clip pick is voiceclip.select = worker.py's clip branch (see voiceclip.py).

  GET /api/voiceclip?w=<weights id>&person=<id>&voice=<same|id|none|upload:hex>&vc=<0|1>&codes=<raw|sam|>
        -> the clip the NEXT run with these settings puts in its system prompt: {kind off|none|codes|upload|error, label, url, s, sha, what}
  GET /api/voiceclip/run/<rid>  -> the same for a run, from its request.json (+ the worker's own summary 'voice' text and sys_block digest)
  GET /api/voiceclip/wav/<name> -> the audible copy (voiceclips/<sha16>.wav = Mimi decode of the exact codes; up-<hex>-<k>.wav = an
                                   upload's exact pre-Mimi waveform). A codes clip not decoded yet is decoded once on CPU (nice, 2 threads).
"""
import asyncio
import json
import os
CANOPY_ROOT = os.environ.get("CANOPY_ROOT") or os.path.expanduser("~")   # cluster home holding ppm/, venvs/, models/ (see README: paths)
HOME_DIR = os.path.expanduser("~")
import re
import subprocess
import time

import numpy as np
from aiohttp import web

import voiceclip as VC

C = {}
PY = os.environ.get("GENWEB_PYTHON", os.sys.executable)
DECODING = {}                                    # sha16 -> Popen
WAVN = re.compile(r"^([0-9a-f]{16}|up-[0-9a-f]{12}-[0-9a-f]{16})\.wav$")


def _jread(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def _decode_async(h, codes):
    p = DECODING.get(h)
    if p is not None and p.poll() is None:
        return
    if len([x for x in DECODING.values() if x.poll() is None]) >= 1:
        return
    npy = f"{VC.CACHE}/{h}.codes.npy"
    np.save(npy, np.asarray(codes, np.int64))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VC_THREADS="2",
               MIMI_DIR=CANOPY_ROOT + "/models/mimi", OMP_NUM_THREADS="2")
    DECODING[h] = subprocess.Popen(["nice", "-n", "15", PY, f"{VC.G}/voiceclip.py", "decode", h, npy], env=env,
                                   stdout=subprocess.DEVNULL, stderr=open(f"{VC.CACHE}/decode-{h}.log", "w"))


def describe(req):
    """request-shaped dict -> page JSON (blocking: file reads; run in an executor)"""
    m = req.get("model")
    if m in ("duplex", "livedup"):
        return {"kind": "na", "label": "duplex run", "reason": "duplex runs: not covered here"}
    try:
        s = VC.select(req)
    except (ValueError, FileNotFoundError) as e:
        return {"kind": "error", "on": True, "label": str(e), "reason": str(e)}
    out = {"kind": s["kind"], "on": s["on"], "label": s["vsrc"], "weights": s["weights"], "person": s["person"]}
    if s["kind"] == "off":
        return dict(out, reason="voice clip prompt is off: no clip in the prompt")
    if s["kind"] == "none":
        return dict(out, reason="voice = none: no clip in the prompt")
    if s["kind"] == "upload":
        import soundfile as sf
        src = s["src"]
        hx = os.path.basename(src)[:-4] if re.match(r"^[0-9a-f]{12}\.wav$", os.path.basename(src)) else None
        if not hx:
            return dict(out, kind="error", reason="upload outside genweb/uploads")
        w = np.ascontiguousarray(s["wave"], np.float32)
        k = __import__("hashlib").sha256(w.tobytes()).hexdigest()[:16]
        name = f"up-{hx}-{k}.wav"
        p = f"{VC.CACHE}/{name}"
        if not os.path.isfile(p):
            sf.write(p + ".tmp.wav", w, 24000, subtype="FLOAT")
            os.replace(p + ".tmp.wav", p)
        return dict(out, url=f"api/voiceclip/wav/{name}", s=round(len(w) / 24000, 2), sha=k,
                    what="the exact -23 LUFS waveform the worker Mimi-encodes (first %d frames)" % s["L"])
    h = s["sha"][:16]
    p = f"{VC.CACHE}/{h}.wav"
    out.update(sha=h, s=round(s["L"] / 12.5, 2), src=os.path.relpath(s["src"], CANOPY_ROOT + ""),
               what="Mimi decode of the exact clip codes in the prompt")
    if os.path.isfile(p):
        return dict(out, url=f"api/voiceclip/wav/{h}.wav")
    _decode_async(h, s["codes"])
    return dict(out, url=None, reason="decoding the clip (a few s)")


async def _off(fn, *a):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *a)


async def next_clip(req):
    q = req.query
    voice = q.get("voice") or ""
    if voice.startswith("upload:"):
        h = voice[7:]
        if not re.match(r"^[0-9a-f]{12}$", h):
            return web.json_response({"kind": "error", "label": "bad upload id", "reason": "bad upload id"})
        voice = f"upload:{C['UPL']}/{h}.wav"
    person = re.sub(r"@orig$", "", q.get("person") or "")
    if voice == "same":
        voice = person
    r = {"mono_weights": q.get("w") or "base", "prefill": {"person": person}, "prompt": {"voice": voice}}
    if q.get("vc") in ("0", "1"):
        r["prompt"]["voice_clip"] = q.get("vc") == "1"
    if q.get("codes") in ("raw", "sam"):
        r["prefill_codes"] = q.get("codes")
    try:
        d = await _off(describe, r)
    except Exception as e:  # noqa: BLE001
        d = {"kind": "error", "label": f"lookup failed: {e}", "reason": str(e)}
    return web.json_response(d, headers={"Cache-Control": "no-store"})


async def run_clip(req):
    rid = req.match_info["rid"]
    if not C["RID"].match(rid):
        raise web.HTTPNotFound()
    rq = _jread(f"{C['RUNS']}/{rid}/request.json")
    if not isinstance(rq, dict):
        raise web.HTTPNotFound()
    try:
        d = await _off(describe, rq)
    except Exception as e:  # noqa: BLE001
        d = {"kind": "error", "label": f"lookup failed: {e}", "reason": str(e)}
    sm = _jread(f"{C['RUNS']}/{rid}/summary.json") or {}
    if sm.get("voice") is not None:                                    # the worker's own words for the clip it used (finished runs)
        d["worker_label"] = sm.get("voice")
        d["label_match"] = sm.get("voice") == d.get("label")
    if isinstance(sm.get("sys_block"), dict):
        d["sys_block"] = sm["sys_block"]
    return web.json_response(d, headers={"Cache-Control": "no-store"})


async def wav(req):
    n = req.match_info["name"]
    if not WAVN.match(n):
        raise web.HTTPNotFound()
    p = f"{VC.CACHE}/{n}"
    if not os.path.isfile(p):
        raise web.HTTPNotFound()
    return web.FileResponse(p, headers={"Cache-Control": "max-age=3600", "Content-Type": "audio/wav"})


def setup(app, g):
    C.update({k: g[k] for k in ("RUNS", "UPL", "RID")})
    os.makedirs(VC.CACHE, exist_ok=True)
    app.router.add_get("/api/voiceclip", next_clip)
    app.router.add_get("/api/voiceclip/run/{rid}", run_clip)
    app.router.add_get("/api/voiceclip/wav/{name}", wav)
