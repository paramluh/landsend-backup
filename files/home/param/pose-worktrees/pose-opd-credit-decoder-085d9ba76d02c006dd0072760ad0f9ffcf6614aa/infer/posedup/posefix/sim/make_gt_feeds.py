#!/usr/bin/env python3
"""posefix GT reference feeds (videort feed format) for the comparison page:
  gt/p01.feed.jsonl            Person 01's own real 12 s (gt/pose_tokens + gt/mimi_local codes; pcm = the B stem is not on disk, so the
                               Mimi codes decoded are NOT done here: pcm = silence, the video's audio conditioning uses the codes)
  gt/c{i}s{k}.feed.jsonl       held-out conv i, side k: B's REAL pose (pdepB_k), B's real Mimi (depB_k) and B's real audio (stem_k), 120 s
usage: python make_gt_feeds.py <out dir>"""
import base64
import json
import os
import sys

import numpy as np
from safetensors.numpy import load_file

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
C = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs"
P01 = "/data/home/amartyavarma/video/demo-data/prompts/01_ab63d745"
CD, SPF = 2, 1920
OFF = np.arange(32) * 2048


def write(name, pose, mimi, audio, prefill_s=0.0):
    with open(f"{OUT}/{name}.feed.jsonl", "w") as fo:
        fo.write(json.dumps({"type": "start", "session": name, "model": "gt", "voice": "01_ab63d745", "fps": 12.5, "person": "01_ab63d745",
                             "prefill_s": prefill_s}) + "\n")
        for k in range(len(pose)):
            seg = audio[k * SPF:(k + 1) * SPF] if audio is not None and len(audio) >= (k + 1) * SPF else np.zeros(SPF, np.float32)
            pcm = np.clip(seg * 32767, -32768, 32767).astype("<i2").tobytes()
            fo.write(json.dumps({"type": "f", "session": name, "f": k, "t": round(k * 0.08, 3), "pose": [int(v) for v in pose[k]],
                                 "mimi": [int(v) for v in mimi[k]], "pcm_b64": base64.b64encode(pcm).decode("ascii")}) + "\n")
        fo.write(json.dumps({"type": "end", "session": name}) + "\n")


pt = np.asarray(load_file(f"{P01}/gt/pose_tokens.safetensors")["codes"], np.int64).T
mm = np.asarray(load_file(f"{P01}/gt/mimi_local.safetensors")["codes"], np.int64).T
write("p01", pt, mm[:len(pt)], None)
for i in range(4):
    z = np.load(f"{C}/c{i}.npz")
    for k in (0,):
        pd, dp, st = z[f"pdepB_{k}"], z[f"depB_{k}"], z[f"stem_{k}"].astype(np.float32)
        pose = pd[CD:] - OFF
        mimi = dp[CD:] - OFF
        n = int(min((pose[:, 0] >= 0).sum(), (mimi[:, 0] >= 0).sum()))
        write(f"c{i}s{k}", pose[:n], mimi[:n], st)
print("ok", os.listdir(OUT))
