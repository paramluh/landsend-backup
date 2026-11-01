#!/usr/bin/env python3
"""posefix FULL PREFILL bundle (pose + words + audio): <out>/<person>.npz = lane [152] (B's word lane over the 12 s chunk, built with the
monoplex repo's infer.lane_from_words = the training rows' placement), audio [32, 150] (gt/mimi_local = lufs23-local Mimi), pose
[32, 150] (gt/pose_tokens). Both runners (replay.py duplex, mono.py monoplex) force lane frames 0..N+1, Mimi + pose code frames 0..N-1.
usage: python make_prefill.py <person> <words.json> <out dir>"""
import os
import sys

MI = "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer"
sys.path.insert(0, MI)
sys.path.insert(0, f"{MI}/ppm")
import numpy as np  # noqa: E402
import infer as MIF  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

person, wpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
pk = f"/data/home/amartyavarma/video/demo-data/prompts/{person}"
words = MIF.read_words(wpath)
lane = np.asarray(MIF.lane_from_words(words, 152), np.int64)
audio = np.asarray(load_file(f"{pk}/gt/mimi_local.safetensors")["codes"], np.int64)
pose = np.asarray(load_file(f"{pk}/gt/pose_tokens.safetensors")["codes"], np.int64)
assert audio.shape == (32, 150) and pose.shape == (32, 150), (audio.shape, pose.shape)
np.savez(f"{out}/{person}.npz", lane=lane, audio=audio, pose=pose)
import ppm_io as IP  # noqa: E402
print(person, "lane words:", " ".join(w["text"] for w in IP.lane_words(lane, 0, 0, 0))[:300])
