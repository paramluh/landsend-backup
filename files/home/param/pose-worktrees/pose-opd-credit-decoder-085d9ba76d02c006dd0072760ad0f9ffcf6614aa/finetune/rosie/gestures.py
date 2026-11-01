"""10-27 user ("the number of gestures with hand movements we have"): hand-gesture strokes in Rosie's REAL training data (her 88 + 4 items in
items.json, pose tokens straight from 24272's stores) with gesturemetric's own definitions (gm.py: per-hand 0.2 s smoothed wrist speed,
peak >= 0.015 shoulder-widths/frame, prominence 0.0075, >= 0.3 s apart, L/R within 0.2 s merged; person-01 frame = gm's real-data
convention, stroke detection is shoulder-width normalised). hands moving = share of 25 fps frames where either wrist >= 0.015.
usage: gestures.py SHARD NSHARD -> out/g_<shard>.json"""
import json
import os
import sys

sys.path.insert(0, "/data/home/amartyavarma/ppm/agents1026/gesturemetric")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gm  # noqa: E402

torch.set_num_threads(2)
sh, ns = int(sys.argv[1]), int(sys.argv[2])
J = json.load(open("/data/home/amartyavarma/ppm/aloe-pose-rosie/items.json"))
its = [("train", x) for x in J["train"]] + [("eval", x) for x in J["eval"]]
SD = "/data/home/amartyavarma/ppm/aloe-pose-ft/stores"
arr = {}
out = []
for i, (sp, (k, r, sw, n, key)) in enumerate(its):
    if i % ns != sh:
        continue
    side = "user" if sw else "assistant"
    if (k, side) not in arr:
        arr[(k, side)] = np.load(f"{SD}/p{k}/rows-00000-of-00001.pose_{side}.npy", mmap_mode="r")
    c = np.asarray(arr[(k, side)][r, :, :n]).T.astype(np.int64)          # [n, 32]
    c = c[(c >= 0).all(1)]
    k55, swp = gm.k55_of(c, "01_ab63d745")
    ws = gm.wrist_speed(k55, swp)
    st, _ = gm.strokes(ws)
    out.append({"split": sp, "key": key, "frames": int(n), "min": round(n / 12.5 / 60, 3), "strokes": int(len(st)),
                "moving_frames": int((ws.max(1) >= gm.PEAK_THR).sum()), "pose_frames": int(len(ws))})
    print(sh, i, key, len(st), flush=True)
os.makedirs("/data/home/amartyavarma/ppm/aloe-pose-rosie/gout", exist_ok=True)
json.dump(out, open(f"/data/home/amartyavarma/ppm/aloe-pose-rosie/gout/g_{sh}.json", "w"))
