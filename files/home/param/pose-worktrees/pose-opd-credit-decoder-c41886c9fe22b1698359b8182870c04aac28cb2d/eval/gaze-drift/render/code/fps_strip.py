"""1 fps head-crop strips (20 tiles, t = 0..19 s): rows = original sample video + each finished GAZE-<E>-<ROW> render.
-> out/<ROW>-face1fps.png (all 20 tiles) and out/<ROW>-face1fps-{a,b}.png (0-9 s / 10-19 s halves, bigger, for reading).
   python fps_strip.py ROW:RID:K [...]"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strips import OUT, R, REQ, VR, face, frames, grid  # noqa: E402

LAB = {"E0": "E0 unedited|re-encode", "E3": "E3 head shape|locked", "E10": "E10 slow orient.|removed",
       "E10h": "E10h REAL pose|20 s ping-pong", "E11": "E11 real body|+ gen head", "E12": "E12 gen body|+ real head",
       "E13": "E13 gen head/torso|+ real arms"}
for spec in sys.argv[1:]:
    row, rid, k = spec.split(":")
    kp = np.load(f"{R}/vdata/kp-{row}.npz")
    rows = [("original", f"{REQ}/{rid}/s{k}/video.mp4", kp["E0"])]
    hy = os.path.exists(f"{R}/vdata/kphy-{row}.npz")
    if hy:   # S-05 / E-03: E10 there is the horizon test (real pose ping-pong), tracks in kppp; hybrids in kphy
        kq, kh = np.load(f"{R}/vdata/kppp-{row}.npz"), np.load(f"{R}/vdata/kphy-{row}.npz")
        var = [("E0", "E0", kp["E0"]), ("E10h", "E10", kq["E10"]), ("E11", "E11", kh["E11"]), ("E12", "E12", kh["E12"]), ("E13", "E13", kh["E13"])]
    else:
        var = [(e, e, kp[e]) for e in ("E0", "E3", "E10") if e in kp]
    for lab, e, tr in var:
        key = f"GAZE-{e}-{row}"
        if os.path.exists(f"{VR}/{key}/DONE"):
            shutil.copy(f"{VR}/{key}/video.mp4", f"{OUT}/{key}.mp4")
            rows.append((LAB[lab], f"{OUT}/{key}.mp4", tr))
    ts = list(range(20))
    for tag, sel, half, sc in (("", ts, 56, 1), ("-a", ts[:10], 60, 2), ("-b", ts[10:], 60, 2)):
        ims = []
        for _, m, track in rows:
            fr = frames(m)
            ims.append([face(fr[min(t * 25, len(fr) - 1)], track[min(t * 25, len(track) - 1), 0, :2], half=half, scale=sc) for t in sel])
        grid(ims, [r[0] for r in rows], sel, f"{row} head crops, 1 fps", lw=130).save(f"{OUT}/{row}-face1fps{tag}.png")
    print(row, [r[0] for r in rows])
