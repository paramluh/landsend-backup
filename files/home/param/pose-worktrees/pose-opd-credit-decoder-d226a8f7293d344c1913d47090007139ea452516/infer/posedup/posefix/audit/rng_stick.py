#!/usr/bin/env python3
"""posefix audit (CPU only): per-codebook frame-to-frame repeat rate + distinct-code count of pose / Mimi tokens in
  (a) live sessions (sessions/*/session.npz: pose, mimi [n, 32] local),
  (b) the offline fast_ppm duo runs (samples/runs/B-*/pose_s*.npy [32, n], generated part = frames >= NPRE),
  (c) real data (gt_pose_s*.npy of the same runs; convs/c*.npz depB_* for Mimi).
Purpose: a frozen / repeating sampling noise inside the CUDA graphs (Gumbel-max multinomial with the same Exp noise every replay) would
show up as near-deterministic repeats in the HIGH codebooks (cb16 / cb31: CE 2.9 / 3.8 nats, i.e. broad distributions that a properly
random sampler re-draws differently almost every frame). Also: PAD calibration of the live text draw (frac PAD vs raw P(PAD) per bin).
usage: python rng_stick.py"""
import glob
import json
import os

import numpy as np

LIVE = "/data/home/amartyavarma/ppm/aloe-posedup-live/sessions"
RUNS = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/runs"
CONVS = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs"
NPRE = 125
CBS = (0, 1, 2, 4, 8, 16, 31)
OFF = np.arange(32) * 2048


def stats(x):
    """x [n, 32] local codes -> {cb: (repeat rate, distinct)}"""
    x = np.asarray(x)
    x = x[(x >= 0).all(1)]
    if len(x) < 3:
        return None
    rep = (x[1:] == x[:-1]).mean(0)
    return {c: (round(float(rep[c]), 3), int(len(np.unique(x[:, c])))) for c in CBS} | {"n": int(len(x))}


def fmt(name, s):
    if s is None:
        return f"{name:44s} (too short)"
    return f"{name:44s} n {s['n']:5d} | " + " ".join(f"cb{c}:{s[c][0]:.2f}/{s[c][1]}" for c in CBS)


out = {}
print("repeat rate / distinct codes per codebook  (cb: repeat/distinct)")
print("--- LIVE sessions (pose, then mimi)")
for d in sorted(glob.glob(f"{LIVE}/*/session.npz")):
    sid = d.split("/")[-2]
    z = np.load(d)
    if "pose" not in z or len(z["pose"]) < 100:
        continue
    j = json.load(open(d.replace(".npz", ".json")))
    pr = j.get("params", {})
    tag = f"{sid} pk{pr.get('pose_topk')} plk{pr.get('pose_light_topk')} cfg{pr.get('pose_cfg_on')}"
    sp, sm = stats(z["pose"]), stats(z["mimi"])
    out[sid] = {"pose": sp, "mimi": sm, "params": {k: pr.get(k) for k in ("pose_temp", "pose_topk", "pose_light_temp", "pose_light_topk",
                                                                        "pose_cfg_on", "light_topk", "cb0_topk")}}
    print(fmt("pose " + tag, sp))
    print(fmt("mimi " + sid, sm))
    # PAD calibration of the lane draw (raw P(PAD) of the cond row vs frac PAD drawn)
    tok, pp = np.asarray(z["tok"]), np.asarray(z["pp"])
    PAD = 151670
    cal = []
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)):
        m = (pp >= lo) & (pp < hi)
        if m.sum() >= 5:
            cal.append(f"[{lo:.2f},{hi:.2f}) n{int(m.sum())} pPAD {pp[m].mean():.2f} fracPAD {(tok[m] == PAD).mean():.2f}")
    print("     PAD calibration: " + " | ".join(cal))
    out[sid]["pad_cal"] = cal
print("--- OFFLINE fast_ppm duo runs (generated part, frames >= NPRE) vs GT of the same rows")
for r in sorted(glob.glob(f"{RUNS}/B-*/")):
    for s in ("s1", "s2"):
        g, t = f"{r}pose_{s}.npy", f"{r}gt_pose_{s}.npy"
        if not (os.path.isfile(g) and os.path.isfile(t)):
            continue
        gp, tp = np.load(g).T, np.load(t).T
        name = r.rstrip("/").split("/")[-1] + "/" + s
        sg, st = stats(gp[NPRE:]), stats(tp[NPRE:])
        sgp = stats(gp[:NPRE])
        out[name] = {"gen": sg, "gt": st, "gen_prefill": sgp}
        print(fmt("gen " + name, sg))
        print(fmt("gt  " + name, st))
print("--- REAL Mimi of B (convs depB, conversation positions)")
for c in sorted(glob.glob(f"{CONVS}/c*.npz")):
    z = np.load(c)
    for k in (0, 1):
        d = z[f"depB_{k}"]
        d = d[(d >= 0).all(1)] - OFF[None]
        out[f"{os.path.basename(c)}/mimi{k}"] = stats(d)
        print(fmt(f"mimi real {os.path.basename(c)} side {k}", stats(d)))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rng_stick.json"), "w"), indent=1, default=str)
