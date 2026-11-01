"""CPU test of chat_ppm: pack plan stats, render + two-track micro of a few packs, and EQUIVALENCE with stage 3's two-track layout
(text2_s3q: lanes = data_1ch.unmerge(merged 1ch ids) placed at P + PREPEND, turn tokens -> PAD). usage: python tests/test_chat_cpu.py"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chat_ppm as C  # noqa: E402

t0 = time.time()
tb, pool, ho = C.table(2)
packs = C.pack_plan(pool, tb, 7500, 20261011)
fill = [sum(min(int(tb[j, 3]) + 1 + int(tb[j, 2]), 7500) for j in p) for p in packs[:20000]]
print(f"split-2: train pool {len(pool):,} -> {len(packs):,} packs of <= 7500 (fill mean {np.mean(fill) / 7500:.3f}, convs/pack {np.mean([len(p) for p in packs]):.2f}); "
      f"held-out {len(ho):,} -> {len(C.pack_plan(ho, tb, 7500, 0, shuffle=False))} packs ({time.time() - t0:.1f}s)", flush=True)
r = C.Renderer(2)
import data_1ch as X  # noqa: E402  (on sys.path via the Renderer)
n_eq = 0
for p in packs[:6]:
    t1 = time.time()
    rows = [r(j) for j in p]
    m = C.chat_micro(rows, p, 7500)
    for row, j in zip(rows, p):                                  # stage-3 equivalence per conversation
        ids1, _ = X.merge_ids(row["a_ids"], row["b_ids"])
        a, b = X.unmerge(ids1)
        lanes, P, _ = C.conv_lanes(row)
        ref = np.full_like(lanes, C.PAD)
        ref[0, P + C.PREPEND:], ref[1, P + C.PREPEND:] = a, b
        ref[np.isin(ref, C.TURN_IDS)] = C.PAD
        ref[1, :P] = row["sys_ids"]
        assert np.array_equal(ref, lanes), j
        assert int(tb[j, 2]) == len(row["a_ids"]) and int(tb[j, 3]) == len(row["sys_ids"]), "table lengths"
        n_eq += 1
    assert (m["y"] == m["ids"][1, m["sel"] + 1]).all() and sum(m["segs"]) == len(m["pid"]) <= 7500
    print(f"chat pack: {len(p)} convs, N {len(m['pid'])}, B-scored {len(m['sel'])}, words {int(m['word'].sum())}, PAD {int(m['pad'].sum())}, "
          f"A words {int((m['ids'][0] < C.CTRL_LO).sum())}, B words {int((m['ids'][1] < C.CTRL_LO).sum())} ({time.time() - t1:.2f}s) {m['prov'][:160]}", flush=True)
tk = r.tok
print("B lane sample:", tk.decode([int(x) for x in m["ids"][1][:400] if x < C.CTRL_LO][:60]).replace("\n", " "))
print("A lane sample:", tk.decode([int(x) for x in m["ids"][0][:400] if x < C.CTRL_LO][:60]).replace("\n", " "))
print(f"CHAT CPU TEST OK ({n_eq} conversations == stage-3 two-track lanes)")
