#!/usr/bin/env python3
"""posefix audit (CPU): onset.py for the MONOPLEX outputs (studio runs with / without pose CFG, infer-test det-p1)."""
import glob
import numpy as np

def win(c, w=125):
    r = (c[1:] == c[:-1]).astype(float)
    return " ".join(f"{r[i:i + w].mean():.2f}" for i in range(0, len(r), w))

S = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
for d in ("runs-studio-nocfg", "runs-studio", "runs-allgen", "runs-poseonly"):
    for f in sorted(glob.glob(f"{S}/{d}/*/codes.npz")):
        z = np.load(f)
        for k in z.files:
            if k in ("pose_s1", "pose", "gt_pose_s1"):
                a = np.asarray(z[k]); a = a if a.shape[0] == 32 else a.T
                print(f"{d}/{f.split('/')[-2]}:{k:10s} {win(a[0])}")
for f in sorted(glob.glob("/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer-test/*/tokens.npz")):
    z = np.load(f)
    print(f"infer-test {f.split('/')[-2]:12s} {win(np.asarray(z['pose_codes'])[0])}")
