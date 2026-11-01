#!/usr/bin/env python3
"""posefix audit (CPU): pose cb0 repeat rate per 10 s window (125 frames) over each long live session + offline duo run: does the
static mode set in at a fixed frame (a buffer / position bug) or at random times (model dynamics)?"""
import glob
import numpy as np

def win(c, w=125):
    r = (c[1:] == c[:-1]).astype(float)
    return " ".join(f"{r[i:i + w].mean():.2f}" for i in range(0, len(r), w))

for d in sorted(glob.glob("/data/home/amartyavarma/ppm/aloe-posedup-live/sessions/*/session.npz")):
    z = np.load(d)
    if "pose" in z and len(z["pose"]) >= 600:
        print(f"live {d.split('/')[-2]:22s} {win(np.asarray(z['pose'])[:, 0])}")
for r in sorted(glob.glob("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/runs/B-*/")):
    for s in ("s1", "s2"):
        p = np.load(f"{r}pose_{s}.npy")[0]; g = np.load(f"{r}gt_pose_{s}.npy")[0]
        print(f"gen  {r.split('/')[-2]}/{s:16s} {win(p)}")
        print(f"gt   {r.split('/')[-2]}/{s:16s} {win(g)}")
