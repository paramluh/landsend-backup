"""GROUND-TRUTH pose + Mimi tokens for the 3 studio packages (10-19): frames 0..24 = the package's llm_inputs prefill_pose / prefill_audio (= the
generated videos' prefill, exactly), frames 25.. = OUR store row's real pose / Mimi codes of the same recording / speaker / window (rows-studio:
the v2pure store row aligned to the package window, offset_frames), as far as that row's contiguous segment reaches (max 750 = 60 s).
-> gt-studio/<identity>.npz (pose [32, N], codes [32, N], n_from_store, source) + gt.json
   python gt_tokens.py      (source aloe-v2pure-ft/aloe_v2pure.env; compute node, CPU)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v2pure-ft/code")
import data_aloe_pose as A  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
PR = "/data/home/amartyavarma/video/demo-data/prompts"
O = f"{W}/gt-studio"
os.makedirs(O, exist_ok=True)
D = A.D
rows = {r["identity"]: r for r in json.load(open(f"{W}/rows-studio/rows.json"))}
sp = A.splits_v2(A.STORES)
bykey = {e[4]: e for s in ("train", "eval") for e in sp[s]}
st = A.Stores(A.STORES)
off = np.arange(D.DEPTH, dtype=np.int64) * D.CBS
out = {}
for idn, r in rows.items():
    li = load_file(f"{PR}/{idn}/llm_inputs.safetensors")
    pa, pp = li["prefill_audio"].astype(np.int64), li["prefill_pose"].astype(np.int64)
    k, rr, sw, n, key = bykey[r["store_row_key"]]
    m = st.conv((k, rr, sw), dict(A.aloe_cfg(), voice_mod=1 << 62, persona_sides="none"), 0)
    P = int(m["P"])
    codes = (np.asarray(m["depth"][1])[P + D.CD:P + D.CD + int(n)] - off).T            # [32, n] the row's real Mimi, item frames
    pose = (np.asarray(m["pdepth"])[P + D.CD:P + D.CD + int(n)] - off).T               # [32, n] the row's real pose
    assert codes.min() >= 0 and pose.min() >= 0 and codes.shape == pose.shape == (32, int(n))
    o = int(r["offset_frames"])                                                         # package frame f = item frame f + o
    N = min(750, int(n) - o)
    gc, gp = np.zeros((32, N), np.int64), np.zeros((32, N), np.int64)
    gc[:, :25], gp[:, :25] = pa, pp
    gc[:, 25:N], gp[:, 25:N] = codes[:, 25 + o:N + o], pose[:, 25 + o:N + o]
    np.savez(f"{O}/{idn}.npz", codes=gc, pose=gp)
    out[idn] = dict(store_row_key=r["store_row_key"], offset_frames=o, frames=N, seconds=round(N / 12.5, 2), item_frames=int(n),
                    source="frames 0..24 = package prefill; 25.. = the store row's real tokens")
    print(idn, json.dumps(out[idn]), flush=True)
json.dump(out, open(f"{O}/gt.json", "w"), indent=1)
