#!/usr/bin/env python3
"""LIVE VOICE DEMO server for the Voxtral-A Aloe fine-tune (aloe-m1vox-ft, job 22335): PPM (Qwen3-32B backbone + 4B speech tower + 0.6B
decoder), channel A = the caller's live mic through the frozen Voxtral Realtime encoder + the run's adapter x s, channel B (words + Mimi)
generated and streamed back. One session at a time on 2 GPUs. = qwen-1ch/infer/s4live-asr/server.py's protocol / session loop / hot swap /
non-preempting test path, with the PPM engine (live_ppm.LivePPM) and the Aloe prompt builder (prompts_ppm).

HTTP  GET /?token=...           the page (index.html, read from disk per request)
      GET /api/presets?token=   held-out Aloe personas / B voice clips, checkpoint, state, control schema
      POST /api/clip?token=     upload a voice clip for B (-> audio-only clip, -23 LUFS, Mimi codes, 3-10 s)
WS    /ws?token=...             client -> server: JSON {"type": "start", "client", "preset", "persona", "clip", params...} | {"type": "set", ...} |
                                {"type": "stop"} | {"type": "ping", "t"}; binary = int16 LE mono 24 kHz mic PCM (any chunking)
                                server -> client: JSON status / ready / per-frame {"type": "f", f, tok, ws, ctl, pp, pn, ms, lag, ps, pd} /
                                settings / done; binary = int16 B audio, 1920 samples per frame
The front proxy (proxy.py) owns the public port and injects the token; the plain URL needs none.
HOT SWAP: a watcher polls the candidate exports (env PPMLIVE_SWAP_TO: comma list, newest COMPLETE marker wins); a newer one is loaded
in-process between sessions (the proxy stays up; the page shows 'model loading').
TEST CLIENTS: only client 'browser' may take over a running session; any other client is refused while a session is active.
No ASR anywhere (standing order): the transcript is B's own word lane.
10-17 (aloe-v2vox hot swap): CK_LABELS.json (re-read per call) gives a checkpoint its page label and prompt mode; prompt 'voice_clip_only' (the
aloe-v2 / v2vox training prompt) = B's voice clip and NO text by default (an empty persona box no longer falls back to the preset's persona);
the run's last step comes from the checkpoint's own train cfg; s from the checkpoint's own run metrics (live_ppm.run_metrics_for); start-up
asserts the v2 prompt selftest (prompts_ppm.selftest_v2) next to the m1vox one.
10-17 ADAPTIVE REPETITION PENALTY (live_ppm.ADAPT_KEYS, live, default off): schema group 'Adaptive repetition penalty' (after the audio
rep-pen rows); every frame message carries spk (B SPEAKING: a word piece in its last speak_window_frames frames), aon (adaptive on), arp / trp
(the audio / text rep-pen in effect that frame); session.npz records spk / arp / trp per frame.
10-17 FACTORISED TEXT SAMPLING (live_ppm module doc; default ON): schema rows factorised_text (bool) + cat_temp at the top of 'Text sampling'
(group hint), every text knob's hint says it acts on the word choice only; min-p default 0.05 (absolute) with factorised, 0 in the joint
fallback (the engine moves min-p to the mode's default on a switch unless it was changed).
10-18 PAD BIAS (live_ppm module doc; live, default 0, range -10..+2): schema row pad_bias 'PAD bias (negative = speak more)' right after
cat_temp at the top of 'Text sampling' (step 0.1); the start-up smoke's all-knobs pass also sets pad_bias -1.
10-24 ADAPTIVE PAD BIAS (live_ppm module doc; live, default off in every profile): schema rows adaptive_pad_on (bool), pad_silent_s (s),
pad_bias_silent, pad_bias_other right after pad_bias in 'Text sampling' (FACT_KEYS); every frame message carries pon (adaptive PAD bias on,
0/1, as used by that frame), sil (seconds since B's last spoken lane token: word piece or NEW_WORD; counts from the first conversation frame)
and pb (the PAD bias in effect that frame, nats: the static pad_bias when off); session.npz records pon / sil / pb per frame; the start-up
smoke's all-knobs pass also turns it on (threshold 1 s, so the 40 silent frames exercise both states)."""
import argparse
import asyncio
import gc
import hmac
import json
import os
import queue
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.environ.get("PPMLIVE_STATE_DIR") or HERE   # 10-17 instance 2: per-instance LOADED.json / SWAP_TO.txt / sessions (default = HERE = instance 1)
sys.path.insert(0, HERE)
import live_ppm as LP  # noqa: E402
import prompts_ppm as PR  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from aiohttp import web, WSMsgType  # noqa: E402

import importlib  # noqa: E402
import a_level as AL  # noqa: E402   (s4live's A-input level rule: -23 LUFS from the first voiced frame + limiter)
AL.LIVE = True

VS = LP.VS
log = LP.log
SPF = VS.SPF24
LOOK = VS.LOOK24
HEARTBEAT_S = 45.0                                 # 10-25 main: 45 s (was 15 s from 10-24; transient upload stalls ended calls); the page pings every 2 s


def ck_meta(ck):
    """10-17: {label, prompt} of a checkpoint from CK_LABELS.json ({abs path: {"label": str, "prompt": "voice_clip_only" | "persona"}}, re-read
    on every call); unlisted checkpoints: no label, prompt 'persona' (= the m1vox prompt: the preset persona text + its clip)."""
    try:
        j = json.load(open(f"{HERE}/CK_LABELS.json"))
    except (OSError, ValueError):
        j = {}
    e = j.get(os.path.abspath(str(ck).rstrip("/")), {}) if ck else {}
    return {"label": e.get("label"), "prompt": e.get("prompt", "persona"), "preset_codes": e.get("preset_codes", "handover"),
            "default_preset": e.get("default_preset"), "extra_presets": e.get("extra_presets")}


def ck_last_step(M):
    """the loaded run's last step: the checkpoint's own trainer cfg 'steps' for an aloe trainer export, else PPMLIVE_LAST_STEP."""
    if M is None:
        return LAST_STEP
    rt, ts = M.info.get("run_tag") or "", M.info.get("total_steps")
    return int(ts) if ts and "aloe" in rt else LAST_STEP
SWAP_TO = [x for x in os.environ.get("PPMLIVE_SWAP_TO", "").split(",") if x]
LAST_STEP = int(os.environ.get("PPMLIVE_LAST_STEP", "366"))
HOST_PARAMS = {
    "a_lufs_on":     (int, 0, 1, 0, "A loudness match on: your mic reaches the model at the -23 LUFS training level from your first voiced words (fast acquisition), then tracks the BS.1770 integrated level; peaks limited to 0.995 (training never exceeded it). 10-17: default OFF (mic audio reaches the model exactly as captured)"),
    "a_target_lufs": (float, -40.0, -10.0, -23.0, "A loudness target (training recipe: -23 LUFS)"),
    "a_slew_db_s":   (float, 0.5, 20.0, 3.0, "max gain change speed (dB/s) after the first ~1 s of voiced speech (which is acquired instantly)"),
    "a_gain_db":     (float, -20.0, 20.0, 0.0, "manual gain on your mic audio (dB), used only when the loudness match is off"),
    "limit_on":         (int, 0, 1, 0, "soft peak limiter on B's audio before int16 on; off (default, 10-17) = bypassed: B's audio goes straight to int16, plain clipping only for out-of-range samples"),
    "limit_ceil":       (float, 0.5, 1.0, 0.98, "soft peak limiter ceiling on B's audio before int16"),
    "limit_attack_ms":  (float, 0.5, 10.0, 2.0, "limiter attack ramp (ms)"),
    "limit_release_ms": (float, 10.0, 500.0, 50.0, "limiter release (ms)"),
}
START_PARAMS = {
    "seed":    (int, 0, 2 ** 31 - 1, 0, "sampling seed (0 = random each Start)"),
    "max_s":   (float, 30.0, 600.0, 600.0, "max session length (s)"),   # 10-25 user: 600 s (10 min = the Aloe training row length; was 300)
    "a_g0_db": (float, -30.0, 30.0, 0.0, "A gain before your first voiced speech (dB). 0 = automatic (the median level of earlier browser mics). From the first voiced frame on, the gain is the estimated -23 LUFS training gain"),
}
HOST_DEFAULTS = {k: v[3] for k, v in HOST_PARAMS.items()}
SHARED_SCHEMA = "/data/home/amartyavarma/qwen-1ch/common/live_params_schema.json"
EXTRA_GROUPS = {"text_rep": "Text sampling", "text_rep_win": "Text sampling",
                "word_greedy": "Text sampling", "a_target_lufs": "A input", "a_slew_db_s": "A input", "a_g0_db": "A input",
                **{k: "Adaptive repetition penalty" for k in LP.ADAPT_KEYS}}
ADAPT_HINT = ("While B is SPEAKING (it emitted a word piece in its last N frames) or SILENT, use that state's repetition penalties instead of "
              "the static ones; off = the static penalties.")
ADAPT_LABELS = {"adaptive_rep_on": "Adaptive repetition penalty", "speak_window_frames": "Speaking window (frames)",
                "audio_rep_speaking": "Audio rep-pen while speaking", "audio_rep_silent": "Audio rep-pen while silent",
                "text_rep_speaking": "Text rep-pen while speaking", "text_rep_silent": "Text rep-pen while silent"}
CFG_NAMES = ("cfg_on", "cfg_s", "cfg_cb0", "cfg_uncond_src")
FACT_KEYS = ("factorised_text", "cat_temp", "pad_bias") + LP.PADAD_KEYS   # 10-17 factorised text sampling (+ 10-18 PAD bias, + 10-24 adaptive PAD bias): own rows at the top of 'Text sampling'
FACT_BOOL = ("factorised_text", "adaptive_pad_on")
FACT_LABELS = {"factorised_text": "Factorised text sampling", "cat_temp": "Category temperature", "pad_bias": "PAD bias (negative = speak more)",
               "adaptive_pad_on": "Adaptive PAD bias", "pad_silent_s": "Silence threshold (s)", "pad_bias_silent": "PAD bias while silent",
               "pad_bias_other": "PAD bias otherwise"}
FACT_STEP = {"cat_temp": 0.05, "pad_bias": 0.1, "pad_silent_s": 0.1, "pad_bias_silent": 0.1, "pad_bias_other": 0.1}
FACT_GROUP_HINT = ("B's text is sampled in two steps (default): first PAD / NEW_WORD / WORD from the model's probabilities (category temperature, PAD bias), "
                   "then, only if WORD, which word piece. Every other text knob here acts on that word choice only; PAD / NEW_WORD timing is never touched by them. "
                   "Adaptive PAD bias: a PAD bias that depends on how long B has been silent (replaces the static PAD bias while on).")
WORD_HINT_KEYS = ("text_temp", "text_topk", "text_topp", "min_p", "min_p_rel", "freq_pen", "text_rep")   # shared-schema rows: this engine's words-only hint
PAGE_SIDE = ("aec", "ns", "agc", "mic_aec", "mic_ns", "mic_agc")
# ONE alias table (= s4live server.py): canonical shared-schema name (v4) -> internal name. Legacy names are accepted inbound too.
ALIAS = {"minp": "min_p", "minp_mode": "min_p_rel", "audio_rep": "audio_rep_pen", "max_len_s": "max_s", "a_gain": "a_gain_db",
         "a_lufs_on": "a_lufs_on", "lim_on": "limit_on", "lim_ceiling": "limit_ceil", "lim_attack_ms": "limit_attack_ms", "lim_release_ms": "limit_release_ms"}
LEGACY = {"min_p": "min_p", "min_p_mode": "min_p_rel", "text_minp": "min_p", "audio_rep_pen": "audio_rep_pen", "max_s": "max_s",
          "a_gain_db": "a_gain_db", "a_lufs_match": "a_lufs_on", "limit_ceil": "limit_ceil", "limit_attack_ms": "limit_attack_ms",
          "limit_release_ms": "limit_release_ms", "cfg_audio": "cfg_cb0", "cfg_strength": "cfg_s", "freq_pen_win": "freq_win",
          "audio_temp": "_audio_temp", "audio_topk": "_audio_topk"}
INV = {v: k for k, v in ALIAS.items()}
CFG_LAT = {"ms": None}                                                 # measured per-frame latency with CFG on (filled by the smoke test)


def _known_internal(k):
    return k in LP.PARAMS or k in HOST_PARAMS or k in START_PARAMS


def canon_in(d):
    """canonical (or legacy) names -> internal names (engine / host / start). Unknown / unsupported names are returned separately."""
    out, unk = {}, {}
    for k, v in d.items():
        if k in ("free_text", "allow_eob"):
            continue                                                    # 10-28 USER ORDER: no lane grammar, ever: legacy key dropped silently
        if k in ("type", "client", "preset", "persona", "starting_context", "clip", "persona_only", "params", "t", "_rejected"):
            continue
        ik = ALIAS.get(k) or LEGACY.get(k) or k
        if ik == "min_p_rel":
            v = 1 if str(v).lower().startswith("rel") or v in (1, True) else 0
        elif ik in ("a_lufs_on", "limit_on"):
            v = 1 if v in (True, 1, "1", "true", "on") else 0
        elif ik in ("cfg_on", "cfg_cb0", "adaptive_rep_on", "factorised_text", "adaptive_pad_on"):
            v = 1 if v in (True, 1, "1", "true", "on") else 0
        if k == "seed":
            out["seed"] = 0 if v in (None, "", "random") else v
        elif k in PAGE_SIDE:
            continue
        elif ik in ("_audio_temp", "_audio_topk"):                    # v1 names -> both audio stages
            for t in (("cb0_temp", "light_temp") if ik == "_audio_temp" else ("cb0_topk", "light_topk")):
                out.setdefault(t, v)
        elif k == "cfg_uncond_src":
            if str(v) in LP.UNCOND_SRC:
                out["cfg_uncond_src"] = str(v)
            else:
                unk[k] = f"uncond source must be one of {list(LP.UNCOND_SRC)}"
        elif _known_internal(ik):
            out[ik] = v
        else:
            unk[k] = "not supported by this demo"
    return out, unk


def canon_out(eng_params, hp, st=None):
    """internal values -> {canonical name: value} (what 'in_effect' reports)."""
    d = {}
    for k, v in eng_params.items():
        if k == "min_p_rel":
            d["minp_mode"] = "relative" if v else "absolute"
        elif k in ("cfg_on", "cfg_cb0", "adaptive_rep_on", "factorised_text", "adaptive_pad_on"):
            d[k] = bool(v)
        else:
            d[INV.get(k, k)] = v
    for k, v in hp.items():
        d[INV.get(k, k)] = bool(v) if k in ("a_lufs_on", "limit_on") else v
    for k, v in (st or {}).items():
        d[INV.get(k, k)] = v
    return d


def _rej(bad):
    out = {}
    for b in bad:
        k = b.split(":")[0].split("=")[0].replace("unknown ", "").strip()
        out[INV.get(k, k)] = b
    return out


def _cfg_note():
    ms = CFG_LAT["ms"]
    if ms is None:
        return "the uncond stream runs every frame in the batch-2 engine (no extra pass)"
    return (f"the uncond stream runs every frame (batch-2 engine): measured {ms:.1f} ms per frame with CFG on"
            + (" (above the 80 ms real-time budget!)" if ms > 80 else " (real time)"))


def ready_schema(eng_params=None, hp=None, st=None):
    """the shared schema's rows (canonical names) with THIS engine's support / defaults / ranges / live flags, + its extra knobs."""
    try:
        shared = json.load(open(SHARED_SCHEMA))["params"]
    except Exception:  # noqa: BLE001
        shared = []
    cur = canon_out(LP.DEFAULTS, HOST_DEFAULTS, dict({k: v[3] for k, v in START_PARAMS.items()}, cfg_uncond_src=LP.UNCOND_SRC[0]))
    rng = {k: [lo, hi] for k, (t, lo, hi) in LP.PARAMS.items()}
    rng.update({k: [v[1], v[2]] for k, v in list(HOST_PARAMS.items()) + list(START_PARAMS.items())})
    rows, covered = [], set()
    for p in shared:
        n = p["name"]
        r = dict(p); r.pop("support", None)
        ik = ALIAS.get(n, n)
        if n in PAGE_SIDE:
            r.update(supported=True, model2_status="page-side (browser getUserMedia constraint); use headphones")
        elif n in ("persona", "voice_clip"):
            r.update(supported=True, live=False)
        elif n == "cfg_cb0":
            r.update(supported=True, live=True, default=bool(cur.get(n)),
                     model2_status="live: with cfg_on, B's cb0 logits = uncond + s (cond - uncond) before the audio rep-pen; one cb0 draw, cb1-31 from the cond decoder; " + _cfg_note())
            covered.add(ik)
        elif n == "cfg_uncond_src":
            r.update(supported=True, live=False, default=LP.UNCOND_SRC[0], choices=list(LP.UNCOND_SRC),
                     hint="What the uncond stream hears through the same Voxtral front + adapter: quiet_frames = s4live's quiet A frames (held-out "
                          "training rows, training level); digital_zero = encoded digital silence. Applies on Start.",
                     model2_status="on Start")
            covered.add("cfg_uncond_src")
        elif n in ("cfg_on", "cfg_s"):
            r.update(supported=True, live=True, default=cur.get(n), range=(rng.get(n) if n == "cfg_s" else None), model2_status="live; " + _cfg_note())
            if n == "cfg_on":
                r["hint"] = (r.get("hint") or "") + " " + _cfg_note() + "."
            covered.add(ik)
        elif _known_internal(ik):
            if n == "lim_ceiling":                                      # 10-17: the limiter's parent tick box, right before its 3 settings
                rows.append({"name": "lim_on", "group": r.get("group") or "B output", "type": "bool", "default": bool(HOST_DEFAULTS["limit_on"]),
                             "range": None, "step": None, "live": True, "label": "Limiter on", "supported": True, "model2_only": True,
                             "hint": "Soft peak limiter on B's audio before int16 (ceiling / attack / release below). Off (default): bypassed, B's "
                                     "audio goes straight to int16 with plain clipping only if a sample is out of range."})
                covered.add("limit_on")
            r.update(supported=True, default=cur.get(n, r.get("default")))
            if ik in WORD_HINT_KEYS:                                    # 10-17: factorised sampling: these act on the word choice only
                r["hint"] = LP.HINTS[ik][:1].upper() + LP.HINTS[ik][1:]
            if ik in rng and n != "minp_mode" and n != "a_lufs_on":
                r["range"] = rng[ik]
            r["live"] = ik not in START_PARAMS
            if n == "a_lufs_on":
                r.update(default=bool(HOST_DEFAULTS["a_lufs_on"]), live=True,
                         hint="Live -23 LUFS loudness match of your mic (the model trained on -23 LUFS audio). Off by default: your mic reaches the "
                              "model exactly as captured (with A gain 0 dB).")
            if n == "a_gain":
                r["hint"] = "Gain on your mic audio before the model hears it (dB), used only when the loudness match is off. 0 dB (default) = exactly as captured."
            if n == "seed":
                r.update(default=None, live=False)
            if n in ("max_len_s",):
                r["range"] = rng["max_s"]
            covered.add(ik)
        else:
            r.update(supported=False, model2_status="not in this model")
        rows.append(r)
    adapt, fact = [], []
    for k, (t, lo, hi) in LP.PARAMS.items():
        if k in covered or k == "min_p_rel":
            continue
        r = {"name": INV.get(k, k), "group": EXTRA_GROUPS.get(k, "Text sampling"), "type": "int" if t is int else "float",
             "default": LP.DEFAULTS.get(k), "range": [lo, hi], "step": 1 if t is int else 0.01, "live": True,
             "hint": LP.HINTS.get(k, ""), "supported": True, "model2_only": True}
        if k in LP.ADAPT_KEYS:                                       # 10-17: own group, right after the audio rep-pen rows
            r.update(label=ADAPT_LABELS[k], group_hint=ADAPT_HINT, model2_status="live")
            if k == "adaptive_rep_on":
                r.update(type="bool", default=bool(LP.DEFAULTS[k]), range=None, step=None)
            adapt.append(r)
            continue
        if k in FACT_KEYS:                                           # 10-17: first rows of 'Text sampling'
            r.update(label=FACT_LABELS[k], group="Text sampling", group_hint=FACT_GROUP_HINT, model2_status="live")
            if k in FACT_BOOL:
                r.update(type="bool", default=bool(LP.DEFAULTS[k]), range=None, step=None)
            else:
                r["step"] = FACT_STEP[k]
            fact.append(r)
            continue
        rows.append(r)
    at = next((i + 1 for i, r in enumerate(rows) if r.get("name") == "audio_rep_win"), len(rows))
    rows[at:at] = adapt
    at = next((i for i, r in enumerate(rows) if r.get("group") == "Text sampling"), 0)
    rows[at:at] = fact
    for k, v in list(HOST_PARAMS.items()) + list(START_PARAMS.items()):
        if k in covered:
            continue
        rows.append({"name": INV.get(k, k), "group": EXTRA_GROUPS.get(k, "A input" if k.startswith("a_") else "Session"), "type": v[0].__name__,
                     "default": v[3], "range": [v[1], v[2]], "step": 0.5 if v[0] is float else 1, "live": k in HOST_PARAMS, "hint": v[4],
                     "supported": True, "model2_only": True})
    return rows


def param_schema():
    """GET /api/params_schema (= s4live): every knob with group, type, range, default, live / start, hint."""
    rows = []
    for k, (t, lo, hi) in LP.PARAMS.items():
        rows.append({"name": k, "type": t.__name__, "min": lo, "max": hi, "default": LP.DEFAULTS.get(k), "live": True,
                     "group": LP.GROUPS.get(k, "other"), "hint": LP.HINTS.get(k, "")})
    for k, (t, lo, hi, dflt, h) in HOST_PARAMS.items():
        rows.append({"name": k, "type": t.__name__, "min": lo, "max": hi, "default": dflt, "live": True, "group": "A input", "hint": h})
    for k, (t, lo, hi, dflt, h) in START_PARAMS.items():
        rows.append({"name": k, "type": t.__name__, "min": lo, "max": hi, "default": dflt, "live": False, "group": "session", "hint": h})
    rows.append({"name": "cfg_uncond_src", "type": "enum", "choices": list(LP.UNCOND_SRC), "default": LP.UNCOND_SRC[0], "live": False,
                 "group": "audio CFG", "hint": "what the CFG uncond stream hears (on Start)"})
    return rows


def _val(spec, d):
    out, bad = {}, []
    for k, v in d.items():
        if k not in spec:
            continue
        t, lo, hi = spec[k][:3]
        try:
            x = t(float(v))
        except (TypeError, ValueError):
            bad.append(f"{k}: not a number"); continue
        if not lo <= x <= hi:
            bad.append(f"{k}={x} outside [{lo}, {hi}] (clamped)"); x = min(max(x, lo), hi)
        out[k] = x
    return out, bad


def validate_host(d):
    return _val(HOST_PARAMS, d)


def validate_start(cfg):
    p = dict(cfg.get("params") or {})
    for k in ("seed", "max_s"):
        if k in cfg and k not in p:
            p[k] = cfg[k]
    st, bad = _val(START_PARAMS, p)
    return {k: st.get(k, START_PARAMS[k][3]) for k in START_PARAMS}, bad


def decode_upload(data):
    """any common audio file -> float32 mono 24 kHz (soundfile, else the bundled ffmpeg) (= s4live)."""
    import io
    import subprocess
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    try:
        w, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        w = w.mean(1)
        if sr != 24000:
            g = gcd(int(sr), 24000)
            w = resample_poly(w, 24000 // g, int(sr) // g).astype(np.float32)
        return w, "soundfile"
    except Exception:  # noqa: BLE001
        import tempfile
        import imageio_ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".upload") as tf:
            tf.write(data); tf.flush()
            r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", tf.name, "-ac", "1", "-ar", "24000",
                                "-f", "f32le", "pipe:1"], capture_output=True, timeout=60)
        if r.returncode != 0 or not r.stdout:
            raise ValueError("could not decode audio: " + r.stderr.decode(errors="replace")[:200])
        return np.frombuffer(r.stdout, np.float32).copy(), "ffmpeg"


class Model:
    """the loaded checkpoint + the engine (graphs captured once) + presets."""

    def __init__(self, a, ck, presets):
        t0 = time.time()
        devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())][:2]
        assert len(devs) == 2, "needs 2 GPUs"
        adapter = a.adapter if (a.adapter and ck == a.ck_init) else None
        self.m, ldev, cfgs, self.info = LP.build_model(ck, devs, adapter=adapter)
        self.eng = LP.LivePPM(self.m, ldev, cfgs, devs, max_s=600.0)   # 10-25: static KV / graph buffers for 600 s (+~1.3 / 1.1 GiB)
        self.eng.capture()
        self.ck, self.step = ck, int(self.info["step"])
        self.tok = LP.IP.tok()
        self.presets = presets
        self.load_s = round(time.time() - t0, 1)
        log(f"MODEL ready: {ck} step {self.step} in {self.load_s}s (s = {self.info['s']:.6g}, {self.info['s_src']})")

    @torch.no_grad()
    def mimi_encode(self, w):
        with torch.cuda.device(self.eng.d0):
            x = torch.as_tensor(np.asarray(w, np.float32), device=self.eng.d0)[None, None]
            return self.eng.mimi.encode(x, num_quantizers=32).audio_codes[0].cpu().numpy().astype(np.int64)

    def free(self):
        self.eng.graphs.clear()
        for k in ("eng", "m"):
            setattr(self, k, None)
        gc.collect()
        torch.cuda.empty_cache()


def upload_codes(Mo, data, clip_min=38, clip_max=125):
    """upload -> mono 24 kHz -> first <= 10 s (training clip_max 125 frames) -> -23 LUFS (whole-clip BS.1770-4 gain, peak 0.995 = the Mimi
    recipe) -> Mimi codes [32, L] (3 s minimum = training clip_min 38 frames)."""
    w, how = decode_upload(data)
    L = min(len(w) // SPF, clip_max)
    if L < clip_min:
        raise ValueError(f"clip too short: {len(w) / 24000:.1f} s (needs >= {clip_min / 12.5:.1f} s)")
    w = w[:L * SPF]
    w, li = VS.lufs_normalize(w)
    codes = Mo.mimi_encode(w)[:, :L]
    assert codes.shape == (32, L), codes.shape
    return codes, w, {"seconds": L / 12.5, "decoder": how, "lufs_in": li.get("lufs_in"), "gain_db": li.get("gain_db"), "lufs_status": li.get("status")}


class Session:
    def __init__(self, srv, ws, loop, cfg):
        self.srv, self.ws, self.loop, self.cfg = srv, ws, loop, cfg
        self.inq = queue.Queue()
        self.outq = asyncio.Queue()
        self.stop = threading.Event()
        self.thread = None
        self.last_rx = time.time()
        self.end_reason = None
        self.ctl = queue.Queue()

    def limit(self, y, hb):
        """soft peak limiter on B's audio (= s4live)."""
        g0 = getattr(self, "_lg", 1.0)
        ceil = float(hb.get("limit_ceil", 0.98))
        pk = float(np.abs(y).max()) if len(y) else 0.0
        tgt = min(1.0, ceil / pk) if pk > 0 else 1.0
        n = len(y)
        if tgt < g0:
            na = max(1, min(n, int(hb.get("limit_attack_ms", 2.0) * 24)))
            ramp = np.concatenate([np.linspace(g0, tgt, na, dtype=np.float32), np.full(n - na, tgt, np.float32)])
            g1 = tgt
        else:
            a = float(np.exp(-n / max(1.0, hb.get("limit_release_ms", 50.0) * 24)))
            g1 = tgt + (g0 - tgt) * a
            ramp = np.linspace(g0, g1, n, dtype=np.float32)
        self._lg = g1
        return np.clip(y * ramp, -ceil, ceil).astype(np.float32)

    def send(self, obj):
        self.loop.call_soon_threadsafe(self.outq.put_nowait, obj)

    def run(self):
        try:
            with self.srv.lock:
                self._run()
        except BaseException as e:  # noqa: BLE001
            log("SESSION ERROR " + traceback.format_exc())
            self.send({"type": "error", "msg": repr(e)[:300]})
        finally:
            self.send(None)

    def prompt(self):
        """-> (prefix, info) from the page's preset / persona text / clip choice."""
        srv, cfg = self.srv, self.cfg
        pres = srv.presets
        p = pres.get(cfg.get("preset")) or pres.items[0]
        mode = ck_meta(srv.M.ck)["prompt"]                            # 10-17: 'voice_clip_only' = the v2 training prompt (no text by default)
        dflt = "" if mode in ("voice_clip_only", "persona_prose") else p["persona"]   # 10-24 persona_prose: no persona text unless the user gives one
        txt = cfg.get("persona")
        txt = dflt if txt is None or not str(txt).strip() else str(txt)
        edited = txt.strip() != dflt.strip()
        cid = str(cfg.get("clip") or ("preset:" + p["id"]))
        clip, csrc = None, None
        if cid.startswith("upload:"):
            up = srv.uploads.get(cid[7:])
            assert up is not None, "uploaded clip not found (re-upload after a server restart)"
            clip, csrc = PR.upload_clip(up["codes"]), {"src": "upload", "lane": False, "seconds": up["seconds"]}
        elif cid.startswith("preset:"):
            cp = pres.get(cid[7:])
            assert cp is not None, ("unknown clip", cid)
            clip, csrc = cp["clip"], {"src": "preset", "lane": False, "speaker": cp["name"], "seconds": round(cp["clip"]["L"] / 12.5, 2),
                                      "src_key": cp["clip"]["src_key"]}
        pre, info = PR.build_prefix(txt, clip, prose=(mode == "persona_prose"))
        info.update(preset=p["id"], persona_edited=edited, clip=csrc, prompt_mode=mode,
                    route=("persona prose (no labels) + voice clip" if mode == "persona_prose" and edited else
                           "custom persona" if edited else ("voice clip only, no text (= the v2 training prompt)" if mode in ("voice_clip_only", "persona_prose")
                                                            else "held-out preset persona")))
        return pre, info, txt

    @torch.no_grad()
    def _run(self):
        Mo = self.srv.M
        cfg = self.cfg
        eng = Mo.eng
        t0 = time.time()
        self.send({"type": "status", "msg": "building system prompt"})
        hp, hbad = validate_host(dict(HOST_DEFAULTS, **{k: v for k, v in (cfg.get("params") or {}).items() if k in HOST_PARAMS}))
        st, sbad = validate_start(cfg)
        self.hp, self.st_params = hp, st
        max_s = min(float(st["max_s"]), eng.max_s)
        pre, pinfo, ptxt = self.prompt()
        P = int(pre["P"])
        seed = int(st["seed"]) if st["seed"] > 0 else int.from_bytes(os.urandom(3), "little") + 1
        st["seed"] = seed
        p0 = {k: v for k, v in (cfg.get("params") or {}).items() if k in LP.PARAMS}
        usrc = str((cfg.get("params") or {}).get("cfg_uncond_src") or LP.UNCOND_SRC[0])
        st["cfg_uncond_src"] = usrc
        self.send({"type": "status", "msg": f"prefilling the system prompt ({P} positions)"})
        eng.start(pre, seed, p0, uncond_src=usrc)
        _, self.param_problems = eng.set_params(p0)
        self.send({"type": "ready", "ck": Mo.ck, "step": Mo.step, "label": ck_meta(Mo.ck)["label"], "P": P, "seed": seed, "a_mimi": 0, "setup_s": round(time.time() - t0, 1),
                   "max_s": max_s, "host_params": hp, "start_params": st, "prompt_route": pinfo.get("route"), "prompt": pinfo,
                   "params": dict(eng.params), "param_problems": self.param_problems, "live_params": True,
                   "schema": ready_schema(), "in_effect": canon_out(eng.params, hp, st),
                   "rejected": dict(_rej(self.param_problems + sbad + hbad), **(cfg.get("_rejected") or {})),
                   "clip": pinfo.get("clip") or {}, "s": Mo.info["s"]})
        log(f"session ready: P {P}, seed {seed}, setup {time.time() - t0:.1f}s, prompt {pinfo}")
        log("session params in effect (" + LP.PROFILE + " defaults): " + json.dumps(canon_out(eng.params, hp, st), sort_keys=True))
        importlib.reload(AL); AL.LIVE = True
        loud = AL.StreamLevel(g0_db=st["a_g0_db"])
        hb = hp
        raw = np.zeros(int((max_s + 5) * 24000) + 10 * SPF, np.float32)
        n_raw = n_full = 0
        f = 0
        T = P
        Tmax = P + int(max_s * 12.5)
        rec = {"a": [], "b": [], "ms": [], "tok": [], "pp": [], "pn": [], "lag": [], "spk": [], "arp": [], "trp": [], "pon": [], "sil": [], "pb": []}
        tok = Mo.tok
        while not self.stop.is_set() and T < Tmax:
            need = (f + 1) * SPF + LOOK
            while n_raw < need and not self.stop.is_set():
                try:
                    x = self.inq.get(timeout=0.5)
                except queue.Empty:
                    if time.time() - self.last_rx > HEARTBEAT_S:
                        self.end_reason = "timeout"
                        self.send({"type": "timeout", "msg": f"no audio or ping for {HEARTBEAT_S:.0f} s"})
                        self.stop.set()
                    continue
                if x is None:
                    self.stop.set(); break
                x = x[:len(raw) - n_raw]
                raw[n_raw:n_raw + len(x)] = x; n_raw += len(x)
            if self.stop.is_set():
                break
            t_ready = time.perf_counter()
            while n_raw - n_full >= SPF:
                loud.target, loud.slew = float(hb["a_target_lufs"]), float(hb["a_slew_db_s"])
                y, _ = loud.push(raw[n_full:n_full + SPF])
                if not hb["a_lufs_on"]:
                    y = (raw[n_full:n_full + SPF] * np.float32(10 ** (hb["a_gain_db"] / 20))).astype(np.float32)
                eng.write_a(n_full, y)
                rec["a"].append(y)
                n_full += SPF
            if n_full < need:
                gdb = loud.g_db if hb["a_lufs_on"] else hb["a_gain_db"]
                eng.write_a(n_full, raw[n_full:need] * np.float32(10 ** (gdb / 20)))
            while not self.ctl.empty():
                upd, unk = self.ctl.get_nowait()
                hu = {k: upd.pop(k) for k in list(upd) if k in HOST_PARAMS}
                clean, bad = eng.set_params({k: v for k, v in upd.items() if k not in START_PARAMS and k != "cfg_uncond_src"})
                if hu:
                    hc, hbad2 = validate_host(hu)
                    hb = dict(hb, **hc); self.hp = hb
                    clean.update(hc); bad += hbad2
                bad += [f"{k}: applies at Start only" for k in upd if k in START_PARAMS or k == "cfg_uncond_src"]
                rej = dict(_rej(bad), **unk)
                self.send({"type": "settings", "f": f, "in_effect": canon_out(eng.params, hb, st),
                           "changed": [INV.get(k, k) for k in clean], "rejected": rej})
                log(f"params at frame {f}: {clean} {bad if bad else ''}")
                self.param_log = getattr(self, "param_log", []) + [{"frame": f, "applied": clean, "problems": bad}]
            dec = eng.step(T)
            tokB, p_pad, p_nw, y = eng.result(T, dec)
            ms = (time.perf_counter() - t_ready) * 1e3
            lag = (n_raw - need) / SPF
            piece, nw = None, False
            if tokB < LP.TE:
                piece = tok.convert_ids_to_tokens(tokB)
                nw = piece.startswith("Ġ")
                piece = tok.decode([tokB])
            self.send({"type": "f", "f": f, "tok": piece, "ws": nw, "ctl": ("PAD" if tokB == LP.PAD else "NW" if tokB == LP.NW else None if tokB < LP.TE else str(tokB)),
                       "pp": round(p_pad, 4), "pn": round(p_nw, 4), "ms": round(ms, 1), "lag": round(lag, 2),
                       "ps": round(eng.last_ptok[0], 4), "pd": round(eng.last_ptok[1], 4),
                       "s": (float(eng.params.get("cfg_s", 1.0)) if eng.params.get("cfg_on") else 1.0),
                       "sa": (float(eng.params.get("cfg_s", 1.0)) if eng.cb0_cfg_live() else 1.0),
                       "cd": round(eng.last_cfgd[0], 4), "tv": round(eng.last_cfgd[1], 4), "tv0": (round(eng.last_cfgd[3], 4) if eng.cb0_cfg_live() else None),
                       "spk": int(eng.last_adapt[0] > 0.5), "aon": int(bool(eng.params.get("adaptive_rep_on"))),
                       "arp": round(eng.last_adapt[1], 4), "trp": round(eng.last_adapt[2], 4),
                       "pon": int(eng.last_padad[0] > 0.5), "sil": round(eng.last_padad[1], 2), "pb": round(eng.last_padad[2], 4)})
            if y is not None:
                if hb.get("limit_on"):
                    y = self.limit(y, hb)
                else:                                                   # 10-17: limiter off (default) = bypassed; int16 clip below only
                    self._lg = 1.0
                self.send(np.clip(y * 32767, -32768, 32767).astype("<i2").tobytes())
                rec["b"].append(y)
            rec["ms"].append(ms); rec["tok"].append(tokB); rec["pp"].append(p_pad); rec["pn"].append(p_nw); rec["lag"].append(lag)
            rec["spk"].append(eng.last_adapt[0]); rec["arp"].append(eng.last_adapt[1]); rec["trp"].append(eng.last_adapt[2])
            rec["pon"].append(eng.last_padad[0]); rec["sil"].append(eng.last_padad[1]); rec["pb"].append(eng.last_padad[2])
            f += 1
            T += 1
        ms = np.asarray(rec["ms"])
        stats = {}
        if len(ms):
            stats = {"frames": int(len(ms)), "p50": round(float(np.percentile(ms, 50)), 2), "p95": round(float(np.percentile(ms, 95)), 2),
                     "p99": round(float(np.percentile(ms, 99)), 2), "max": round(float(ms.max()), 2), "mean": round(float(ms.mean()), 2),
                     "rtf": round(float(ms.mean()) / 80.0, 3)}
        words = " ".join(w["text"] for w in LP.IP.lane_words(np.asarray(rec["tok"]))) if rec["tok"] else ""
        sid = time.strftime("%Y%m%d-%H%M%S")
        od = f"{self.srv.a.sessions}/{sid}"
        os.makedirs(od, exist_ok=True)
        np.savez_compressed(f"{od}/session.npz", a=np.concatenate(rec["a"]) if rec["a"] else np.zeros(0), b=np.concatenate(rec["b"]) if rec["b"] else np.zeros(0),
                            ms=ms, tok=np.asarray(rec["tok"]), a_vec=eng.vrec[:f].float().cpu().numpy().astype(np.float16),
                            pp=np.asarray(rec["pp"]), pn=np.asarray(rec["pn"]), lag=np.asarray(rec["lag"]),
                            spk=np.asarray(rec["spk"], np.float32), arp=np.asarray(rec["arp"], np.float32), trp=np.asarray(rec["trp"], np.float32),
                            pon=np.asarray(rec["pon"], np.float32), sil=np.asarray(rec["sil"], np.float32), pb=np.asarray(rec["pb"], np.float32))
        json.dump({"ck": Mo.ck, "step": Mo.step, "label": ck_meta(Mo.ck)["label"], "s": Mo.info["s"], "seed": seed, "params": dict(eng.params), "param_log": getattr(self, "param_log", []),
                   "cfg": {k: v for k, v in self.cfg.items() if k != "type"}, "prompt": pinfo, "persona": ptxt, "latency_ms": stats, "B_words": words,
                   "P": P, "end_reason": self.end_reason}, open(f"{od}/session.json", "w"), indent=1, default=str)
        self.send({"type": "done", "latency": stats, "session": sid})
        log(f"session {sid} done: {stats} | {words[:200]}")


class Server:
    def __init__(self, a):
        self.a = a
        self.token = open(a.token_file).read().strip()
        self.lock = threading.Lock()                   # held by a running session and by a model (re)load
        self.active = None
        self.M = None
        self.swap_to = None
        self.state = "loading"
        self.uploads = {}
        self.presets = None
        self.prefix_selftest = None

    def load(self, ck):
        self.state = "loading " + "/".join(ck.rstrip("/").split("/")[-2:])
        if self.M is not None:
            self.M.free(); self.M = None
        codes, first, extra = ck_meta(ck)["preset_codes"], ck_meta(ck)["default_preset"], ck_meta(ck)["extra_presets"]   # 10-24/25: follow the checkpoint
        if self.presets is None or (self.presets.codes, self.presets.first, self.presets.extra) != (codes, first, extra):
            self.presets = PR.Presets(codes=codes, first=first, extra=extra)
            log(f"PRESETS codes={codes}: {[(p['id'], p['name'], p['clip']['src_key']) for p in self.presets.items]}")
            self.prefix_selftest = self.presets.selftest()
            log(f"PREFIX SELFTEST {'PASS' if self.prefix_selftest['PASS'] else 'FAIL'} {json.dumps(self.prefix_selftest)}")
            assert self.prefix_selftest["PASS"], "prompt builder != the trainer's podcast_prefix_vox"
            self.prefix_selftest_v2 = PR.selftest_v2(self.presets)
            log(f"PREFIX SELFTEST V2 {'PASS' if self.prefix_selftest_v2['PASS'] else 'FAIL'} {json.dumps(self.prefix_selftest_v2)}")
            assert self.prefix_selftest_v2["PASS"], "voice-clip-only prompt builder != the v2 trainer's prefix_v2"
        self.M = Model(self.a, ck, self.presets)
        self.smoke()
        self.state = "ready"
        json.dump({"ck": ck, "step": self.M.step, "label": ck_meta(ck)["label"], "prompt_mode": ck_meta(ck)["prompt"], "last_step": ck_last_step(self.M),
                   "loaded": time.strftime("%F %T %Z"), "info": self.M.info, "load_s": self.M.load_s,
                   "smoke": self.smoke_res}, open(f"{STATE}/LOADED.json", "w"), indent=1, default=str)

    @torch.no_grad()
    def smoke(self):
        """internal Start smoke (preset 0, its clip, a few frames of silence through the graphed step) before the model is offered."""
        eng = self.M.eng
        p0 = self.presets.items[0]
        pm = ck_meta(self.M.ck)["prompt"]
        pre, info = PR.build_prefix("" if pm == "voice_clip_only" else p0["prose"] if pm == "persona_prose" else p0["persona"], p0["clip"],
                                    prose=(pm == "persona_prose"))
        eng.start(pre, 1234)
        ms = []
        for f in range(12):
            eng.write_a(f * SPF, np.zeros(SPF + LOOK, np.float32))
            t1 = time.perf_counter()
            dec = eng.step(eng.P + f)
            tok, pp, pn, y = eng.result(eng.P + f, dec)
            ms.append((time.perf_counter() - t1) * 1e3)
            assert np.isfinite(pp) and np.isfinite(pn) and (y is None or np.isfinite(y).all())
        eng.start(pre, 1234, {"cfg_on": 1, "cfg_s": 2.0, "cfg_cb0": 1, "audio_rep_pen": 1.1, "text_rep": 1.1, "freq_pen": 0.3, "min_p": 0.05,
                              "adaptive_rep_on": 1, "audio_rep_speaking": 1.1, "audio_rep_silent": 1.3, "text_rep_speaking": 1.1, "text_rep_silent": 1.2,
                              "pad_bias": -1.0, "adaptive_pad_on": 1, "pad_silent_s": 1.0, "pad_bias_silent": -2.0, "pad_bias_other": 0.5})
        msc = []
        for f in range(40):
            eng.write_a(f * SPF, np.zeros(SPF + LOOK, np.float32))
            t1 = time.perf_counter()
            dec = eng.step(eng.P + f)
            tok, pp, pn, y = eng.result(eng.P + f, dec)
            msc.append((time.perf_counter() - t1) * 1e3)
            assert np.isfinite(pp) and np.isfinite(pn) and (y is None or np.isfinite(y).all())
        CFG_LAT["ms"] = float(np.percentile(msc[3:], 95))
        self.smoke_res = {"frames": 12, "ms_p50": round(float(np.median(ms[2:])), 1), "ms": [round(x, 1) for x in ms], "P": eng.P,
                          "cfg_on_all_knobs_ms_p50": round(float(np.median(msc[3:])), 1), "cfg_on_ms_p95": round(CFG_LAT["ms"], 1),
                          "all_knobs_incl_adaptive_rep": True, "all_knobs_incl_adaptive_pad": True}
        log(f"SMOKE PASS {self.smoke_res}")

    def latest_swap(self):
        """the newest COMPLETE candidate: env PPMLIVE_SWAP_TO + the lines of SWAP_TO.txt (re-read every poll: the fallback watcher appends the
        node-local step-366 copy there if the consolidated /data export never appears)."""
        best = None
        extra = []
        try:
            extra = [ln.strip() for ln in open(f"{STATE}/SWAP_TO.txt") if ln.strip() and not ln.startswith("#")]
        except OSError:
            pass
        for d in SWAP_TO + extra:
            if os.path.isfile(f"{d}/COMPLETE"):
                k = os.path.getmtime(f"{d}/COMPLETE")
                if best is None or k > best[0]:
                    best = (k, d)
        return best[1] if best else None

    def watcher(self):
        while True:
            time.sleep(20)
            try:
                ck = self.latest_swap()
                if ck and self.M is not None and os.path.abspath(ck) != os.path.abspath(self.M.ck) and ck != getattr(self, "_failed", None):
                    if self.swap_to != ck:
                        log(f"HOT SWAP pending: {ck}")
                    self.swap_to = ck
                if self.swap_to and self.active is None:
                    with self.lock:
                        ck, self.swap_to = self.swap_to, None
                        old = self.M.ck if self.M is not None else None
                        try:
                            self.load(ck)
                            log(f"HOT SWAP done: {old} -> {ck}")
                        except Exception:  # noqa: BLE001
                            log("HOT SWAP FAILED, reloading the previous checkpoint: " + traceback.format_exc())
                            self._failed = ck
                            self.M = None
                            gc.collect(); torch.cuda.empty_cache()
                            self.load(old)
            except Exception:  # noqa: BLE001
                log("watcher: " + traceback.format_exc())

    def ok(self, req):
        return hmac.compare_digest(req.query.get("token", ""), self.token)

    async def index(self, req):
        if not self.ok(req):
            return web.Response(status=403, text="forbidden")
        return web.FileResponse(f"{HERE}/index.html", headers={"Cache-Control": "no-store"})

    async def presets_api(self, req):
        if not self.ok(req):
            return web.Response(status=403, text="forbidden")
        M, pr = self.M, self.presets
        return web.json_response({"state": self.state, "ck": M and M.ck, "step": M and M.step, "busy": self.active is not None,
                                  "presets": pr.public(prose=ck_meta(M and M.ck)["prompt"] == "persona_prose") if pr else [], "clips": pr.clips_public() if pr else [],
                                  "default_clip": pr.default_clip() if pr else {}, "default_preset": pr.items[0]["id"] if pr else None,
                                  "swap_pending": self.swap_to, "start_name": "", "upload_ok": True, "last_step": ck_last_step(M),
                                  "label": M and ck_meta(M.ck)["label"], "prompt_default": ck_meta(M and M.ck)["prompt"],
                                  "s": M and M.info["s"], "schema": ready_schema(), "defaults_profile": LP.PROFILE})

    async def clip(self, req):
        if not self.ok(req):
            return web.Response(status=403, text="forbidden")
        if self.M is None:
            return web.json_response({"error": "model loading"}, status=503)
        data = await req.read()
        if not data or len(data) > 25 << 20:
            return web.json_response({"error": "empty or > 25 MB"}, status=400)
        try:
            codes, w, info = await asyncio.get_running_loop().run_in_executor(None, upload_codes, self.M, data)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)[:300]}, status=400)
        import secrets
        cid = secrets.token_hex(6)
        self.uploads[cid] = {"codes": codes, **info}
        od = f"{self.a.sessions}/uploads"
        os.makedirs(od, exist_ok=True)
        np.savez_compressed(f"{od}/{cid}.npz", codes=codes, wav=w.astype(np.float16))
        return web.json_response({"id": "upload:" + cid, **info, "label": f"your upload, {info['seconds']:.1f} s (audio-only clip)"})

    async def ws(self, req):
        if not self.ok(req):
            return web.Response(status=403, text="forbidden")
        ws = web.WebSocketResponse(max_msg_size=8 << 20, heartbeat=20)
        await ws.prepare(req)
        loop = asyncio.get_running_loop()
        sess = None
        sender = None

        async def pump(s):
            while True:
                item = await s.outq.get()
                if item is None:
                    break
                try:
                    if isinstance(item, bytes):
                        await ws.send_bytes(item)
                    else:
                        await ws.send_str(json.dumps(item))
                except Exception:  # noqa: BLE001
                    break
            if self.active is s:
                self.active = None

        try:
            async for msg in ws:
                if sess is not None:
                    sess.last_rx = time.time()
                if msg.type == WSMsgType.BINARY:
                    if sess is not None:
                        sess.inq.put(np.frombuffer(msg.data, "<i2").astype(np.float32) / 32768.0)
                elif msg.type == WSMsgType.TEXT:
                    d = json.loads(msg.data)
                    if d.get("type") == "ping":
                        await ws.send_str(json.dumps({"type": "pong", "t": d.get("t")}))
                    elif d.get("type") == "start":
                        if self.state != "ready" or self.M is None:
                            await ws.send_str(json.dumps({"type": "error", "msg": f"model {self.state}; try again in about a minute"}))
                            continue
                        old = self.active
                        if old is not None and d.get("client") != "browser":   # automated / unknown clients never preempt anyone
                            await ws.send_str(json.dumps({"type": "error", "msg": "busy: a session is active (automated clients never take over)"}))
                            await ws.close(code=1013, message=b"busy")
                            break
                        if old is not None:                                      # TAKEOVER (browser only): the newest browser Start wins
                            log("session taken over by a new browser session")
                            old.end_reason = "taken_over"
                            old.send({"type": "taken_over", "msg": "a newer session took over"})
                            old.stop.set(); old.inq.put(None)
                            if old.thread is not None:
                                await loop.run_in_executor(None, old.thread.join, 60)
                            if self.active is old:
                                self.active = None
                        pin, unk = canon_in(dict(d, **(d.get("params") or {})))
                        d = dict(d, params=pin, _rejected=unk)
                        sess = Session(self, ws, loop, d)
                        self.active = sess
                        sender = asyncio.ensure_future(pump(sess))
                        sess.thread = threading.Thread(target=sess.run, daemon=True)
                        sess.thread.start()
                    elif d.get("type") in ("set", "params") and sess is not None:
                        pin, unk = canon_in(dict(d, **(d.get("params") or {})))
                        sess.ctl.put((pin, unk))
                    elif d.get("type") == "stop" and sess is not None:
                        sess.inq.put(None)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            if sess is not None:
                sess.stop.set(); sess.inq.put(None)
                if sender is not None:
                    try:
                        await asyncio.wait_for(sender, 30)
                    except Exception:  # noqa: BLE001
                        pass
                if self.active is sess:
                    self.active = None
        return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--adapter", default="", help="VOX_ADAPTER dir: used only when --ck is the PPM init (test path: init + adapter + s init)")
    ap.add_argument("--port", type=int, default=8785)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--sessions", default=f"{STATE}/sessions")
    a = ap.parse_args()
    a.ck_init = a.ck if a.adapter else None
    os.makedirs(a.sessions, exist_ok=True)
    srv = Server(a)
    ck = srv.latest_swap() or a.ck                                    # a COMPLETE swap target at start-up wins over --ck
    try:
        srv.load(ck)
    except Exception:  # noqa: BLE001
        if ck == a.ck:
            raise
        log(f"load of {ck} failed, falling back to --ck {a.ck}: " + traceback.format_exc())
        srv._failed = ck
        srv.M = None; gc.collect(); torch.cuda.empty_cache()
        srv.load(a.ck)
    threading.Thread(target=srv.watcher, daemon=True).start()
    app = web.Application(client_max_size=26 << 20)
    app.router.add_get("/", srv.index)
    app.router.add_get("/api/presets", srv.presets_api)
    app.router.add_get("/api/params_schema", lambda r: web.json_response(param_schema()) if srv.ok(r) else web.Response(status=403, text="forbidden"))
    app.router.add_get("/ws", srv.ws)
    app.router.add_post("/api/clip", srv.clip)
    app.router.add_get("/healthz", lambda r: web.json_response({"state": srv.state, "busy": srv.active is not None, "ck": srv.M and srv.M.ck}))
    host = os.environ.get("PPMLIVE_BIND", "127.0.0.1")              # 10-25: 0.0.0.0 for a backend another node's front proxy routes to (token-checked)
    log(f"serving on {host}:{a.port}")
    web.run_app(app, host=host, port=a.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
