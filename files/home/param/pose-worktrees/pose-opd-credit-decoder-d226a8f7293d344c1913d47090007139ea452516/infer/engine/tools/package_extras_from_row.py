#!/usr/bin/env python3
"""Make a prompt package's two LLM extras from an Aloe-format row whose conversation frame 0 = the package's frame 0:
  words.json                   [[start_s, end_s, text], ...] decoded from the row's B lane (start = the word's first frame / 12.5; end = start)
  voice_clip_mimi.safetensors  {'codes': [32, L]} = the voice clip of the row's system block (the training rule's clip)
and check that infer.py's lane builder reproduces the row's lane from words.json exactly (frames 0 .. --check-frames).
CPU only.   python tools/package_extras_from_row.py ROW.pkl OUT_DIR [--check-frames 752]"""
import argparse
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ppm"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import infer  # noqa: E402
import ppm_io as IP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("row")
ap.add_argument("out")
ap.add_argument("--check-frames", type=int, default=None)
a = ap.parse_args()
x = pickle.load(open(a.row, "rb"))
P = int(x["P"])
lane = np.asarray(x["ids"][1, P:], np.int64)
n = len(lane) if a.check_frames is None else int(a.check_frames)
ws = IP.lane_words(lane[:n], 0, 0, None)
words = [[round(w["frame"] / IP.FPS, 4), round(w["frame"] / IP.FPS, 4), w["text"]] for w in ws]
rebuilt = infer.lane_from_words(words, n)
eq = rebuilt == lane[:n]
os.makedirs(a.out, exist_ok=True)
json.dump({"words": words, "source": f"lane of {a.row} (row_key {x.get('row_key')})", "lane_rebuild_exact_frames": [0, n] if eq.all() else None},
          open(f"{a.out}/words.json", "w"), indent=0)
cpos = np.flatnonzero(x["depth"][1, :P, 0] >= 0)
assert len(cpos), "the row's system block has no voice clip"
clip = (x["depth"][1, cpos] - IP.OFF).T
from safetensors.numpy import save_file  # noqa: E402
save_file({"codes": np.ascontiguousarray(clip.astype(np.int64))}, f"{a.out}/voice_clip_mimi.safetensors")
print(json.dumps({"words": len(words), "lane_frames_checked": n, "lane_rebuild_exact": bool(eq.all()), "mismatch_frames": np.flatnonzero(~eq)[:10].tolist(),
                  "voice_clip_frames": int(clip.shape[1]), "clip_positions": [int(cpos[0]), int(cpos[-1]) + 1]}))
