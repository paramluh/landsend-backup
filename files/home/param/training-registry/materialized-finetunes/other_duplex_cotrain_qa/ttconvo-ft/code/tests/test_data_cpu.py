"""CPU test of data_ppm on a node holding the stores (s3-rows, pose-s3-samdata). Checks invariants + prints a few rows' prefixes.
usage: python tests/test_data_cpu.py [n_speech_rows]"""
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_ppm as D  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
cfg = D.default_cfg()
t0 = time.time()
node = D.R3.load_node("/scratch/amartyavarma/s3-rows")
idx = D.DS.load_index("/scratch/amartyavarma/s3-rows")
plan = D.speech_plan(node, idx, 1, 8, 0, 0.0005, "ppm-v1", 2)
print(f"node {node['node']} rows {len(node['length'])} full {(node['length'] == 7500).sum()} stripe(local0) {len(plan['rows'])} eval {plan['eval_rows']} "
      f"({time.time() - t0:.1f}s)", flush=True)
ds = D.SpeechDS("/scratch/amartyavarma/s3-rows", plan["rows"][:N], cfg, 0)
cnt = collections.Counter()
Ps, sws, cuts = [], [], 0
t0 = time.time()
for i in range(N):
    m = ds[i]
    pv = json.loads(m["prov"])
    T = len(m["pid"])
    assert m["ids"].shape == (2, T) and m["depth"].shape == (2, T, 32) and sum(m["segs"]) == T
    assert (m["y"] == m["ids"][1, m["sel"] + 1]).all() and (m["word"] | m["pad"] | m["eob"] | m["nw"]).all()
    assert m["ids"].max() <= D.MAX_ID and (m["light"] >= 0).all() and (m["cb0"][m["light"]] >= 0).all()
    assert T == 7500 + D.CD, (T, pv["P"])
    assert (m["present"][:, pv["P"]:] == 1).all() and (m["present"][:, :pv["P"]].sum(0) == 1).all()
    Ps.append(pv["P"]); sws.append(m["swap"]); cuts += pv["persona_cut"] > 0
    cnt["voice_drawn"] += pv["voice_drawn"]; cnt["clips"] += len(pv["clips"]); cnt["persona_blocks"] += len(pv["persona"])
    cnt["empty"] += int(pv["P"] == 5)
    if i < 4 or (pv["clips"] and cnt["shown_clip"] < 2):
        cnt["shown_clip"] += bool(pv["clips"])
        print(json.dumps({k: pv[k] for k in ("row_key", "swap", "chan_B_data_side", "P", "voice_drawn", "clips", "persona", "persona_cut")}))
        print(D.PT.decode_prefix(m["ids"], m["present"], pv["P"], D._tok_init())[:600].replace("\n", " | "))
print(f"{N} speech micros in {time.time() - t0:.1f}s: {dict(cnt)}; P min/med/max {min(Ps)}/{int(np.median(Ps))}/{max(Ps)}; swap frac {np.mean(sws):.2f}; "
      f"rows with persona cut {cuts}", flush=True)
# pose
pn = D.PS.load_node("/scratch/amartyavarma/pose-s3-samdata/rows")
pp = D.PS.plan(pn, 20261010, 8, 0, 1, 8, 0, 1)
pds = D.PoseDS(pn, pp["rows"][:4], cfg, 0)
for i in range(2):
    m = pds[i]
    N_ = len(m["pid"])
    assert m["pdepth"].shape == (N_, 32) and (m["present"][0] == 0).all() and (m["y"] == m["ids"][1, m["sel"] + 1]).all()
    print(f"pose micro: N {N_} segs {len(m['segs'])} sel {len(m['sel'])} cb0 {(m['cb0'] >= 0).sum()} light {len(m['light'])} pcb0 {(m['pcb0'] >= 0).sum()} "
          f"plight {len(m['plight'])}")
# text
tds = D.TextDS("/data/home/amartyavarma/data/gemma-txtdup-v1", "train", [0, 1, 2], dict(cfg, txt_cap=7000, pack_attn="segments"))
for i in range(3):
    m = tds[i]
    assert (m["y"] == m["ids"][1, m["sel"] + 1]).all()
    print(f"text micro: N {len(m['pid'])} convs {len(m['segs'])} B-scored {len(m['sel'])} words {m['word'].sum()} prov {m['prov'][:120]}")
print("DATA CPU TEST OK")
