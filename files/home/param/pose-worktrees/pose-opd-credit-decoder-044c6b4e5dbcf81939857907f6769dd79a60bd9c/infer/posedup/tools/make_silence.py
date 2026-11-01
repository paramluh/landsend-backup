#!/usr/bin/env python3
"""10-25 POSE CFG: the Mimi codes of digital silence for live_ppm_pose's pose-CFG uncond row, computed EXACTLY as fast_ppm.FastPPM.__init__
(IP.Mimi(dl).encode(np.zeros((p_max + n_fr) * 1920))[:, :p_max + n_fr]) for the server's engine sizes (p_max 2000, max_s 600 -> n_fr 7508).
  python tools/make_silence.py <out .npy> [p_max] [max_s]      (GPU; writes <out> + <out>.json)"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import live_ppm_pose as LP  # noqa: E402

out = sys.argv[1]
p_max = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
max_s = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
n_fr = int(round(max_s * LP.FPS)) + 8                              # = LivePPM.n_fr
n = p_max + n_fr
t0 = time.time()
with torch.no_grad():
    sil = LP.IP.Mimi(torch.device("cuda:0")).encode(np.zeros(n * 1920, np.float32))[:, :n]
assert sil.shape == (LP.DEPTH, n) and sil.dtype == np.int64 and sil.min() >= 0, (sil.shape, sil.dtype)
np.save(out, sil)
info = {"shape": list(sil.shape), "p_max": p_max, "max_s": max_s, "n_fr": n_fr, "call": "IP.Mimi(cuda).encode(np.zeros(n * 1920))[:, :n] (= fast_ppm)",
        "mimi": LP.IP.MIMI, "sha256": hashlib.sha256(sil.tobytes()).hexdigest(), "frame0": sil[:, 0].tolist(), "frame100": sil[:, 100].tolist(),
        "distinct_frames": int(len({tuple(c) for c in sil.T.tolist()})), "s": round(time.time() - t0, 1)}
json.dump(info, open(out + ".json", "w"), indent=1)
print(json.dumps(info))
