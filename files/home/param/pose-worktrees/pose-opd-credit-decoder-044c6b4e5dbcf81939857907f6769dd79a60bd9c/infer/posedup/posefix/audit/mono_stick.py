#!/usr/bin/env python3
"""posefix audit (CPU): the same per-codebook repeat-rate / distinct stats (rng_stick.py) on MONOPLEX (aloe-pose-mono-8020 export-357) pose
outputs: samples/aloe-posemono-poseonly runs-* (codes.npz) + gt-studio, and aloe-pose-monoplex-infer-test outputs (tokens.npz)."""
import glob
import json

import numpy as np

CBS = (0, 1, 2, 4, 8, 16, 31)


def stats(x):
    x = np.asarray(x)
    x = x[(x >= 0).all(1)]
    if len(x) < 3:
        return None
    rep = (x[1:] == x[:-1]).mean(0)
    return {c: (round(float(rep[c]), 3), int(len(np.unique(x[:, c])))) for c in CBS} | {"n": int(len(x))}


def fmt(name, s):
    if s is None:
        return f"{name:60s} (none)"
    return f"{name:60s} n {s['n']:5d} | " + " ".join(f"cb{c}:{s[c][0]:.2f}/{s[c][1]}" for c in CBS)


S = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
for f in sorted(glob.glob(f"{S}/runs-*/*/codes.npz")) + sorted(glob.glob(f"{S}/gt-studio/*.npz")):
    z = np.load(f)
    keys = list(z.files)
    for k in keys:
        a = np.asarray(z[k])
        if a.ndim == 2 and 32 in a.shape and "pose" in k.lower():
            a = a if a.shape[1] == 32 else a.T
            m = json.load(open(f.replace("codes.npz", "meta.json"))) if f.endswith("codes.npz") else {}
            npf = int(m.get("prefill_frames", m.get("prompt_frames", 0)) or 0)
            print(fmt(f"{f.split('poseonly/')[1]}:{k} (skip {npf})", stats(a[npf:])))
    if not any("pose" in k.lower() for k in keys):
        print(f, "keys", keys)
for f in sorted(glob.glob("/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer-test/*/tokens.npz")) + \
        sorted(glob.glob("/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer-test/*/*/tokens.npz")):
    z = np.load(f)
    p = np.asarray(z["pose_codes"]).T
    npf = int(z["prefill_frames"]) if "prefill_frames" in z.files else 0
    print(fmt(f"{f.split('infer-test/')[1]} (skip {npf})", stats(p[npf:])))
