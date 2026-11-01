#!/usr/bin/env python3
"""gesturemetric PER-RUN score for the playground (sits next to run_stats.py's gestures/min): one genweb run dir -> JSON.
Same inputs / clock as gen_extract.py (tokens.npz 'pose' from conversation code frame 0, words.json ch B = the pose owner's model word
lane, b.wav; measured part = t >= pose_prefill_frames x 0.08 s), same definitions as gm.py:
  gesture_timing_kappa   (hit - chance) / (1 - chance): strokes within +-0.2 s of an own word onset vs circularly shifted words
                         (0 = flailing / unrelated to the speech; real Aloe 18 s clips ~0.28, see ref)
  gesture_timing_loud    the same against loudness peaks of b.wav
  strokes_per_min, timing_hits / timing_n
  substance              stroke rate on substantive words / on fillers + backchannels (pooled over runs only: too few words per run)
A single 20 s run has ~5-15 strokes, so its kappa is noisy (real 18 s clips p10-p90 in ref); compare settings over several runs.
CPU only (~1-2 s per 20 s run after the tokenizer loads).
usage: python gm_run.py RUN_DIR [RUN_DIR ...]   -> one JSON line per run"""
import json
import os
import sys
import time

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import gm  # noqa: E402
import gen_extract as GX  # noqa: E402

ANALYSIS = os.path.join(H, "out", "analysis.json")


def ref():
    """real / shuffled 18 s clip references from out/analysis.json (re-read per call: the file may be regenerated)."""
    try:
        a = json.load(open(ANALYSIS))["real"]
    except (OSError, ValueError, KeyError):
        return None
    out = {}
    for k in ("clip_real", "clip_shuf_conv"):
        s = a.get(k) or {}
        out[k] = {e: {"kappa": (s.get(e) or {}).get("kappa"), "kappa_p10_p90": (s.get(e) or {}).get("kappa_p10_p90")}
                  for e in ("words", "loud")}
        out[k]["substance"] = (s.get("substance") or {}).get("lift")
    return out


def run_gesture(run_dir):
    t0 = time.time()
    run_dir = os.path.normpath(run_dir)
    summ = GX.jload(os.path.join(run_dir, "summary.json"), {}) or {}
    req = GX.jload(os.path.join(run_dir, "request.json"), {}) or {}
    words = GX.jload(os.path.join(run_dir, "words.json"), []) or []
    person = GX.person_of(summ, req)
    st, _, _, L, _, _ = GX.strokes_of(np.load(os.path.join(run_dir, "tokens.npz"))["pose"], person)
    lo = int(summ.get("pose_prefill_frames") or summ.get("prefill_frames") or 0) * 0.08
    own = [(float(w["t0"]), float(w["t1"]), str(w["w"])) for w in words if w.get("ch", "B") == "B"]
    m = (st >= lo) & (st <= L)
    out = {"run": os.path.basename(run_dir), "person": person, "measured_s": round(L - lo, 1), "strokes": int(m.sum()),
           "strokes_per_min": round(60 * float(m.sum()) / max(L - lo, 1e-9), 1)}
    tw = gm.timing(st, gm.word_events(own, "words"), lo, L, seed=0) if L - lo >= 2 * gm.SHIFT_MIN + 1 else {"kappa": None}
    out["gesture_timing_kappa"] = None if tw.get("kappa") is None else round(tw["kappa"], 3)
    out["timing_hits"], out["timing_n"] = tw.get("hits"), tw.get("n")
    wav = os.path.join(run_dir, "b.wav")
    out["gesture_timing_loud"] = None
    if os.path.exists(wav) and L - lo >= 2 * gm.SHIFT_MIN + 1:
        import soundfile as sf
        x, sr = sf.read(wav, dtype="float32", always_2d=True)
        env = gm.loudness_env(x.mean(1), sr)
        tl = gm.timing(st, gm.loud_events(env), lo, min(L, len(env) / 100.0), seed=0)
        out["gesture_timing_loud"] = None if tl.get("kappa") is None else round(tl["kappa"], 3)
    rows = gm.substance_rows(st, own, lo, L)
    out["substance_rows"] = {"sub": [sum(1 for c, _ in rows if c == "sub"), sum(s for c, s in rows if c == "sub")],
                             "fill": [sum(1 for c, _ in rows if c == "fill"), sum(s for c, s in rows if c == "fill")]}
    (ns, hs), (nf, hf) = out["substance_rows"]["sub"], out["substance_rows"]["fill"]
    out["substance_lift"] = round((hs / ns) / (hf / nf), 3) if ns and nf and hf else None   # per run: very noisy, pool over runs
    out["note"] = "one 20 s run is noisy: compare settings over >= 10 runs (real 18 s clips p10-p90 in ref)"
    out["ref"] = ref()
    out["scorer"] = "agents1026/gesturemetric/gm_run.py (gm.py definitions)"
    out["secs"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":
    import torch
    torch.set_grad_enabled(False)
    for d in sys.argv[1:]:
        print(json.dumps(run_gesture(d)), flush=True)
