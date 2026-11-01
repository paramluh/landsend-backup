"""absolute head / root drift and yaw offsets per window (GT vs prefill vs generated), per set; plus per-clip yaw drift table."""
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, "/data/home/amartyavarma/ppm/gaze-drift/codebooks")
import cb_lib as L  # noqa: E402

torch.set_num_threads(8)
OUT = "/data/home/amartyavarma/ppm/gaze-drift/codebooks"
gt, gen = L.load_all()
W = {"prefill": (0, 50), "matched": (50, 300), "late": (300, 500)}


def m(pos, f0):
    sm = (pos[:, 5] + pos[:, 6]) / 2
    nose = pos[:, 0]
    ft = L.feats(pos, f0["es"], f0["ss"])
    yaw = ft["yaw"]
    pre_y, pre_sm, pre_n = yaw[:50].mean(), sm[:50].mean(0), nose[:50].mean(0)
    o = {}
    for w, (lo, hi) in W.items():
        if hi > len(pos):
            continue
        lpy = gaussian_filter1d(yaw[lo:hi], 12, mode="nearest")
        o[w] = {"yaw_mean_abs_off": float(np.mean(np.abs(lpy - pre_y))),
                "yaw_end_off": float(abs(yaw[hi - 50:hi].mean() - pre_y)),
                "yaw_lf": float(np.std(lpy)),
                "yaw_hf": L.jitter(yaw[lo:hi]),
                "root_end_off_sw": float(np.linalg.norm(sm[hi - 50:hi].mean(0) - pre_sm) / f0["ss"]),
                "nose_end_off_sw": float(np.linalg.norm(nose[hi - 50:hi].mean(0) - pre_n) / f0["ss"]),
                "nose_rel_end_off_ear": float(np.linalg.norm((nose - sm)[hi - 50:hi].mean(0) - (nose - sm)[:50].mean(0)) / f0["es"]),
                "head_rel_lf_ear": L.lowpass_std(ft["head"][lo:hi]),
                "head_rel_hf_ear": L.jitter(ft["head"][lo:hi]),
                "conf_head": None}
    return o


res = {}
per_clip = {}
for row in L.ROWS:
    r3g = L.dec_codes(gt[row][None])[0]
    f0 = L.feats(L.positions(r3g))
    srcs = {"GT": r3g}
    for s in L.SETS:
        srcs[s] = L.dec_codes(gen[s][row][None])[0]
    for s, r3 in srcs.items():
        o = m(L.positions(r3), f0)
        for w in o:
            o[w]["conf_head"] = float(r3[W[w][0]:W[w][1], :5, 2].mean())
        res.setdefault(s, []).append(o)
        per_clip.setdefault(row, {})[s] = {w: round(o[w]["yaw_end_off"], 3) for w in o}
agg = {}
for s, lst in res.items():
    agg[s] = {}
    for w in W:
        v = [d[w] for d in lst if w in d]
        if v:
            agg[s][w] = {k: round(float(np.mean([x[k] for x in v])), 4) for k in v[0]}
R = {"note": "video frames: prefill 0-50, matched 50-300, late 300-500. yaw offsets vs the clip's own prefill mean yaw (proxy (nose_x - ear_mid_x)/ear_dx); "
             "root/nose end offsets = |mean of last 2 s of window - prefill mean| in shoulder widths (absolute frame position); "
             "nose_rel = nose relative to shoulder midpoint, ear-distance units; lf = std of 0.5 s low-pass, hf = RMS of residual vs 2-frame gauss.",
     "agg": agg, "per_clip_yaw_end_off": per_clip}
import fcntl  # noqa: E402
RJ = f"{OUT}/results.json"
with open(RJ + ".lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    A = json.load(open(RJ))
    A["abs_drift"] = R
    tmp = f"{RJ}.{os.getpid()}.tmp"
    json.dump(A, open(tmp, "w"), indent=1)
    os.replace(tmp, RJ)
for s in agg:
    for w in agg[s]:
        print(s, w, agg[s][w], flush=True)
