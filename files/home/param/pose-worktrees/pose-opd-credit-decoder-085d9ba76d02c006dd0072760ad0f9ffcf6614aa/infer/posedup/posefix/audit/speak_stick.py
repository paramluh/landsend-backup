#!/usr/bin/env python3
"""posefix audit (CPU): pose cb0 repeat rate split by B SPEAKING vs LISTENING (B's lane holds a word piece / NEW_WORD within +-12 frames of
the pose frame), for live sessions (tok per step frame f, pose per code frame f - 2), the offline duo runs (lanes.npz + pose_s*.npy,
generated part) and real data (convs laneB_d / gtpose_d)."""
import glob

import numpy as np

TE, NW = 151643, 151671
W = 12


def speaking(lane):
    s = (lane < TE) | (lane == NW)
    k = np.convolve(s.astype(float), np.ones(2 * W + 1), "same") > 0
    return k


def split(cb0, spk):
    r = cb0[1:] == cb0[:-1]
    s = spk[1:]
    a = r[s].mean() if s.sum() > 20 else np.nan
    b = r[~s].mean() if (~s).sum() > 20 else np.nan
    return a, b, s.mean()


rows = []
for d in sorted(glob.glob("/data/home/amartyavarma/ppm/aloe-posedup-live/sessions/*/session.npz")):
    z = np.load(d)
    if "pose" not in z or len(z["pose"]) < 300:
        continue
    tok, pose = np.asarray(z["tok"]), np.asarray(z["pose"])
    n = len(pose)
    spk = speaking(tok)[:n]                                      # pose row i = code frame cf = i = the lane frame it describes
    a, b, fs = split(pose[:n, 0], spk[:n])
    rows.append(("live " + d.split("/")[-2], a, b, fs))
R = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/runs"
for r in sorted(glob.glob(f"{R}/B-*/")):
    L = np.load(f"{r}lanes.npz")
    for k, s in enumerate(("s1", "s2")):
        p = np.load(f"{r}pose_{s}.npy")[0]
        g = np.load(f"{r}gt_pose_{s}.npy")[0]
        lane = np.asarray(L[f"lane_{s}"])
        spk = speaking(lane)[:len(p)]
        a, b, fs = split(p[125:], spk[125:len(p)])
        rows.append((f"gen  {r.split('/')[-2]}/{s}", a, b, fs))
for c in sorted(glob.glob("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c*.npz")):
    z = np.load(c)
    for k in (0, 1):
        lane, g = np.asarray(z[f"laneB_{k}"]), np.asarray(z[f"gtpose_{k}"])[0]
        n = min(len(g), len(lane))
        a, b, fs = split(g[:n], speaking(lane)[:n])
        rows.append((f"real {c.split('/')[-1]} side {k}", a, b, fs))
print(f"{'source':40s} cb0-repeat SPEAKING | LISTENING | frac speaking")
for name, a, b, fs in rows:
    print(f"{name:40s} {a:9.3f} | {b:9.3f} | {fs:.2f}")
