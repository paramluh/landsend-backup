#!/usr/bin/env python3
"""10-28 user ("how do i know what the clip you are using for cloning is i want to be able to hear it"): which voice clip a monoplex run
puts in its system prompt, and an audible copy of it. No torch at import (the page server imports this).

select(req) = worker.py Worker.run's clip branch, line for line (same files, same order, same crops), for a request.json-shaped dict:
  off      voice clip prompt off (request prompt.voice_clip false)            -> no clip
  none     voice none / empty                                                  -> no clip
  upload   lufs23(load_wav24(path))[:L * 1920], L = min(len // 1920, 125)      -> the EXACT waveform the worker hands to Mimi encode
  codes    v2pure rule clip | <openings>/<p>/voice_clip | gt/mimi_local[:, :125] -> the EXACT int64 codes the worker puts in system_block
The weights config (prompt / openings / voice) is resolved as worker.py resolves it at start for GENWEB_WEIGHTS=<id>. Codes clips are
deterministic files (no per-run randomness, no seed). Audible copy of a codes clip = the Mimi decode (kyutai/mimi, CPU fp32) of exactly
those codes, cached as voiceclips/<sha256(int64 C bytes)[:16]>.wav; an upload's copy = its pre-Mimi waveform (float32 wav, bit-exact).
CLI (compute node only): voiceclip.py decode-all | decode <sha16> <codes .npy>"""
import hashlib
import json
import os
CANOPY_ROOT = os.environ.get("CANOPY_ROOT") or os.path.expanduser("~")   # cluster home holding ppm/, venvs/, models/ (see README: paths)
HOME_DIR = os.path.expanduser("~")
import sys

import numpy as np

G = os.path.dirname(os.path.abspath(__file__))
CACHE = f"{G}/voiceclips"
PEOPLE = [os.environ.get("GENWEB_PEOPLE", CANOPY_ROOT + "/video/demo-data/prompts"), os.environ.get("GENWEB_ACTORS", CANOPY_ROOT + "/video/dmd-rt/actors/pkg"),
          os.path.dirname(os.path.abspath(__file__)) + "/people_extra"]   # = worker.py PEOPLE
SPF = 1920


def person_file(person, rel):                                         # = worker.py person_file
    for r in PEOPLE:
        if os.path.isfile(f"{r}/{person}/{rel}"):
            return f"{r}/{person}/{rel}"
    raise FileNotFoundError(f"person {person}: no {rel}")


def lufs23(w):                                                         # = worker.py lufs23 (verbatim)
    try:
        import pyloudnorm as pyln
        L = pyln.Meter(24000).integrated_loudness(w.astype(np.float64))
        g = (-23.0 - L) if np.isfinite(L) else 0.0
    except Exception:  # noqa: BLE001   (no pyloudnorm: RMS-based fallback ~ -23 LUFS for speech)
        r = float(np.sqrt(np.mean(w ** 2)) + 1e-9)
        g = -20.0 - 20 * np.log10(r)
    y = w * np.float32(10 ** (g / 20))
    pk = float(np.abs(y).max()) if len(y) else 0
    return (y * (0.995 / pk) if pk > 0.995 else y).astype(np.float32)


def load_wav24(path):                                                  # = worker.py load_wav24 (verbatim)
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != 24000:
        g = gcd(int(sr), 24000)
        w = resample_poly(w, 24000 // g, int(sr) // g).astype(np.float32)
    return w


def wcfg(weights):
    """weights id -> (WEIGHTS, BASE_PROMPT, OPEN_DIR, W_VOICE) as worker.py's module level sets them for GENWEB_WEIGHTS=weights"""
    bp, od, wv = "mono8020", "openings", "own"
    if weights == "base" and os.path.isfile(f"{G}/MONO_CK.json"):
        bp = json.load(open(f"{G}/MONO_CK.json")).get("prompt", "mono8020")
    if weights not in ("base", "rosie") and os.path.isfile(f"{G}/weights/{weights}.json"):
        wc = json.load(open(f"{G}/weights/{weights}.json"))
        bp, od, wv = wc.get("prompt", "mono8020"), wc.get("openings", "openings"), wc.get("voice", "own")
    return weights, bp, od, wv


def codes_sha(c):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(c, np.int64)).tobytes()).hexdigest()


def select(req):
    """request.json-shaped dict -> {kind off|none|upload|codes, vsrc (= the worker's summary 'voice' text), on, codes | wave, src, sha}"""
    from safetensors.numpy import load_file
    weights = str(req.get("mono_weights") or "base")
    WEIGHTS, BASE_PROMPT, od, wv = wcfg(weights)
    pr = dict(req.get("params") or {})
    vcl = pr.pop("voice_clip", None)
    vcl = (req.get("prompt") or {}).get("voice_clip", req.get("voice_clip", vcl))
    vc_on = True if vcl is None else vcl in (True, 1, "1", "true")
    pf = req.get("prefill") or {}
    person = pf.get("person") or "01_ab63d745"
    prm = req.get("prompt") or {}
    voice = prm.get("voice")
    pcodes = req.get("prefill_codes") if req.get("prefill_codes") in ("raw", "sam") else None
    if pcodes:
        od, wv = ("openings", "own") if pcodes == "raw" else ("openings-sam", "opening_clip")
    dbg = req.get("debug_inputs") if req.get("hidden") is True else None
    if isinstance(dbg, dict) and dbg.get("openings") in ("openings", "openings-sam"):
        od, wv = dbg["openings"], dbg.get("voice") or ("own" if dbg["openings"] == "openings" else "opening_clip")
    out = {"weights": weights, "person": person, "on": vc_on}
    if not vc_on:
        return dict(out, kind="off", vsrc="no voice clip (voice clip prompt off)")
    if voice in (None, "", "none"):
        return dict(out, kind="none", vsrc="none")
    if str(voice).startswith("upload:"):
        path = str(voice)[7:]
        w = lufs23(load_wav24(path))
        L = min(len(w) // SPF, 125)
        if L < 38:
            raise ValueError(f"uploaded clip too short: {len(w) / 24000:.1f} s (needs >= 3 s)")
        return dict(out, kind="upload", wave=w[:L * SPF], src=path, vsrc=f"upload ({L / 12.5:.1f} s, -23 LUFS)", L=L)
    vp = person if voice == "same" else str(voice)
    rc = f"{G}/v2pure-clips/{vp}.safetensors"
    if BASE_PROMPT == "v2pure" and WEIGHTS == "base" and pcodes != "sam" and os.path.isfile(rc):
        clip, src = np.asarray(load_file(rc)["codes"], np.int64), rc
        vsrc = f"{vp} v2pure rule clip ({clip.shape[1] / 12.5:.1f} s)"
    elif wv == "opening_clip" and not os.path.isfile(f"{G}/{od}/{vp}/voice_clip.safetensors"):
        raise ValueError(f"no {od} voice clip for {vp}: pick an actor with SAM data or upload a clip")
    elif wv == "opening_clip":
        src = f"{G}/{od}/{vp}/voice_clip.safetensors"
        clip = np.asarray(load_file(src)["codes"], np.int64)
        vsrc = f"{vp} {od} voice clip ({clip.shape[1] / 12.5:.1f} s)"
    else:
        src = person_file(vp, "gt/mimi_local.safetensors")
        clip = np.asarray(load_file(src)["codes"], np.int64)[:, :125]
        vsrc = f"{vp} own voice (10 s)"
    return dict(out, kind="codes", codes=clip, src=src, vsrc=vsrc, sha=codes_sha(clip), L=int(clip.shape[1]))


def all_clip_sources():
    """every codes clip any monoplex run can select today -> [(src, codes int64)]"""
    import glob
    from safetensors.numpy import load_file
    out = []
    for f in sorted(glob.glob(f"{G}/v2pure-clips/*.safetensors")) + sorted(glob.glob(f"{G}/openings*/*/voice_clip.safetensors")):
        out.append((f, np.asarray(load_file(f)["codes"], np.int64)))
    seen = set()
    for r in PEOPLE:
        for d in sorted(os.listdir(r)) if os.path.isdir(r) else []:
            if d in seen:
                continue
            try:
                f = person_file(d, "gt/mimi_local.safetensors")
            except FileNotFoundError:
                continue
            seen.add(d)
            out.append((f, np.asarray(load_file(f)["codes"], np.int64)[:, :125]))
    return out


def _mimi():
    import torch
    from transformers import MimiModel
    torch.set_num_threads(int(os.environ.get("VC_THREADS", "4")))
    d = os.environ.get("MIMI_DIR") or CANOPY_ROOT + "/models/mimi"
    return torch, MimiModel.from_pretrained(d).eval()


def decode_to(m, torch, codes, path):
    import soundfile as sf
    with torch.no_grad():
        c = torch.as_tensor(np.asarray(codes, np.int64))[None]
        w = m.decode(c).audio_values[0, 0].float().numpy()[:c.shape[-1] * SPF]
    tmp = path + ".tmp.wav"
    sf.write(tmp, w.astype(np.float32), 24000, subtype="FLOAT")
    os.replace(tmp, path)


def main():
    os.makedirs(CACHE, exist_ok=True)
    if sys.argv[1] == "decode-all":
        torch, m = _mimi()
        idx = {}
        for src, c in all_clip_sources():
            h = codes_sha(c)[:16]
            p = f"{CACHE}/{h}.wav"
            if not os.path.isfile(p):
                decode_to(m, torch, c, p)
            idx[h] = {"src": src, "shape": list(c.shape), "sha256": codes_sha(c)}
            print(h, src, c.shape, flush=True)
        json.dump(idx, open(f"{CACHE}/INDEX.json.tmp", "w"), indent=0)
        os.replace(f"{CACHE}/INDEX.json.tmp", f"{CACHE}/INDEX.json")
    elif sys.argv[1] == "decode":
        h, npy = sys.argv[2], sys.argv[3]
        c = np.load(npy)
        assert codes_sha(c)[:16] == h, "codes / sha mismatch"
        torch, m = _mimi()
        decode_to(m, torch, c, f"{CACHE}/{h}.wav")


if __name__ == "__main__":
    main()
