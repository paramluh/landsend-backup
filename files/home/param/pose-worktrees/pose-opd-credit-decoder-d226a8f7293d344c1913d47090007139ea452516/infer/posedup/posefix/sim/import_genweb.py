#!/usr/bin/env python3
"""posefix: import finished playground runs genweb/runs/posefix[b]-p01_* (monoplex, full prefill) into out/fullM/<name>.{npz,json} in the
replay / mono.py format, and their avatar renders into clips/fullM/<name>.mp4. posefixb-* (prefill seconds corrected: N = 2 x round(s / 0.16)
+ 2, = the duplex runs) replace posefix-* of the same name. usage: python import_genweb.py"""
import glob
import json
import os
import shutil

import numpy as np
import soundfile as sf

G = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/runs"
P = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix"
best = {}
for d in sorted(glob.glob(f"{G}/posefix-p01_*")) + sorted(glob.glob(f"{G}/posefixb-p01_*")):
    name = os.path.basename(d).split("-", 1)[1]
    if os.path.isfile(f"{d}/summary.json") and os.path.isfile(f"{d}/tokens.npz"):
        sm = json.load(open(f"{d}/summary.json"))
        want = {"2": 26, "5": 64, "10": 126, "1168": 148}[name.split("_full")[1].split("_")[0]]
        if int(sm.get("pose_prefill_frames", -1)) == want:
            best[name] = d
n = 0
for name, d in best.items():
    sm = json.load(open(f"{d}/summary.json"))
    z = np.load(f"{d}/tokens.npz")
    b, sr = sf.read(f"{d}/b.wav", dtype="float32")
    assert sr == 24000, sr
    N = int(sm["pose_prefill_frames"])
    pose = z["pose"]
    gen = pose[N:, 0].astype(int)
    words = json.load(open(f"{d}/words.json"))
    np.savez_compressed(f"{P}/out/fullM/{name}.npz", pose=pose, mimi=z["mimi"], tok=z["lane"], b=b)
    json.dump({"name": name, "model": "monoplex export-357 (playground worker)", "person": sm.get("person"), "seed": sm.get("seed"),
               "pose_prefill_frames": N, "prefill_streams": sm.get("prefill_streams"), "params": sm.get("params"), "src": d,
               "gen_cb0_change": float((gen[1:] != gen[:-1]).mean()) if len(gen) > 1 else None, "gen_cb0_distinct": int(len(np.unique(gen))),
               "words": " ".join(w["w"] for w in words)[:400]}, open(f"{P}/out/fullM/{name}.json", "w"), indent=1)
    if os.path.isfile(f"{d}/avatar.mp4") and not os.path.isfile(f"{P}/clips/fullM/{name}.mp4"):
        os.makedirs(f"{P}/clips/fullM", exist_ok=True)
        shutil.copyfile(f"{d}/avatar.mp4", f"{P}/clips/fullM/{name}.mp4")
    n += 1
print(f"imported {n} of {len(best)} finished runs")
