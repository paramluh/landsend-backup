"""PSNR between two mp4s (mean over common frames, prefill 0-2 s, after 2 s).  python pair_psnr.py A.mp4 B.mp4"""
import sys
import numpy as np
sys.path.insert(0, "/data/home/amartyavarma/ppm/gaze-drift/render/code")
from strips import frames, psnr  # noqa: E402
a, b = frames(sys.argv[1]), frames(sys.argv[2])
n = min(len(a), len(b))
p = [psnr(a[i], b[i]) for i in range(n)]
print(f"n {n} all {np.mean(p):.2f} prefill {np.mean(p[:50]):.2f} after2s {np.mean(p[50:]):.2f}")
