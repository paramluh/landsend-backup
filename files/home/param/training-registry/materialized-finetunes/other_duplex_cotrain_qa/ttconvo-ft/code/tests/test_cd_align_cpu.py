"""Code-delay alignment test (user decision 10-11: CD = POSE_DELAY = PPM_CD, default 2). On a node with the stores. Checks, with the
configured CD (asserted == 2 unless PPM_CD says otherwise):
  speech rows   every Mimi frame f of each channel lands EXACTLY CD positions after its text frame: depth[c, P + CD + f] == codes_c[:, f]
                (+ codebook offsets), positions P .. P + CD - 1 carry no conversation codes, the lane frame f sits at P + f (word pieces
                on their frame); cb0 labels start at P + CD; the row is cropped to 7,500 + CD positions (D23).
  voice clips   inside the system prompt: clip frame f's codes at slot b0 + 1 + f + CD (b0 = the <sos> slot), exactly CD PAD-lane tail
                slots, no code on <eos> / the footer.
  pose rows     Mimi AND pose of row frame s0 + q - CD at segment position q (the same delay), text lane of frame s0 + q at q.
  packed rows   one segment, pid 0..N-1 (plain causal; user decision 10-11).
usage: python tests/test_cd_align_cpu.py [n_rows]"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_ppm as D  # noqa: E402

CD = D.CD
assert CD == int(os.environ.get("PPM_CD", "2")) and D.SPR.TAIL == CD and D.PS.CD == CD and D.DS.CD == CD, (CD, D.SPR.TAIL)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
cfg = D.default_cfg()
assert cfg["pose_delay"] == CD and cfg["crop_conv"]
off = np.arange(32, dtype=np.int64) * 2048
node = D.R3.load_node("/scratch/amartyavarma/s3-rows")
idx = D.DS.load_index("/scratch/amartyavarma/s3-rows")
plan = D.speech_plan(node, idx, 1, 8, 0, 0.0005, "ppm-v1", 1)
ds = D.SpeechDS("/scratch/amartyavarma/s3-rows", plan["rows"][:N], cfg, 0)
n_clip = n_rows = 0
for i in range(N):
    m = ds[i]
    pv = json.loads(m["prov"])
    row = ds.st[int(ds.rows[i])]
    P, T, sw = pv["P"], len(m["pid"]), m["swap"]
    assert T == 7500 + CD, T
    nc = 7500 - P
    for c in (0, 1):
        side = ("user", "assistant")[c ^ sw]                         # model channel c holds data side (c ^ swap)
        codes = np.asarray(row[f"codes_{side}"])[:, :nc].astype(np.int64)
        d = m["depth"][c]
        assert np.array_equal(d[P + CD:P + CD + nc], codes.T + off), ("Mimi frame f not at P + CD + f", i, c)
        assert (d[P:P + CD] < 0).all() and (d[P + CD + nc:] < 0).all()
    assert (m["cb0"][:P + CD] < 0).all() and (m["cb0"][P + CD:P + CD + nc] >= 0).all()
    # text frame f at P + f: every word's first piece sits on its start frame (or after the previous piece)
    side_b = ("user", "assistant")[1 ^ sw]
    w = row[f"words_{side_b}"]
    pids = np.asarray(row[f"piece_ids_{side_b}"]); po = np.asarray(row[f"piece_offsets_{side_b}"])
    for j in range(min(len(w), 50)):
        f = int(np.floor(float(w[j]["start"]) * 12.5 + 1e-6))
        if f + 1 >= nc:
            break
        lane = m["ids"][1, P + f:P + f + 40]
        assert int(pids[po[j]]) in lane.tolist(), ("first piece of a word not at/after its frame", i, j)
    # voice clips in the prefix
    for k, cl in enumerate(pv["clips"]):
        c = "AB".index(cl["track"])
        b0 = [p for p in range(P) if m["ids"][c, p] == (D.SPR.SOSA if c == 0 else D.SPR.SOSB)][0]
        L = cl["frames"][1] - cl["frames"][0]
        dd = m["depth"][c, b0:b0 + 1 + L + CD + 1]
        assert (dd[:1 + CD] < 0).all(), "codes before clip frame 0 + CD"
        assert (dd[1 + CD:1 + CD + L] >= 0).all(), "clip codes not at b0 + 1 + f + CD"
        assert (dd[1 + CD + L:] < 0).all(), "a code on <eos> / after the clip"
        lane = m["ids"][c, b0:b0 + 1 + L + CD + 1]
        assert lane[-1] == (D.SPR.EOSA if c == 0 else D.SPR.EOSB) and (lane[1 + L:1 + L + CD] == D.PAD_ID).all(), "CD PAD tail slots then <eos>"
        n_clip += 1
    n_rows += 1
print(f"speech rows: {n_rows} rows, {n_clip} voice clips aligned at CD={CD}; crop to {7500 + CD} positions OK")
# pose rows (Mimi and pose at the same delay)
pn = D.PS.load_node("/scratch/amartyavarma/pose-s3-samdata/rows")
pp = D.PS.plan(pn, 1, 8, 0, 1, 8, 0, 1)
pr = D.PS.PoseRows(pn)
for i in range(4):
    row = pr[int(pp["rows"][i])]
    m0 = D.pose_micro_ppm(row, cfg, 0)
    a = 0
    for (s0, L) in D.PS.row_segments(row):
        q = np.arange(L)
        ok = q >= CD
        mim = np.asarray(row["codes"])[:, (s0 + q - CD)[ok]].T.astype(np.int64)
        pos = np.asarray(row["pose"])[:, (s0 + q - CD)[ok]].T.astype(np.int64)
        dm, dp = m0["depth"][1][a:a + L][ok], m0["pdepth"][a:a + L][ok]
        assert np.array_equal(dm, np.where(mim >= 0, mim + off, -1)) and np.array_equal(dp, np.where(pos >= 0, pos + off, -1)), ("pose / Mimi delay", i)
        assert (m0["depth"][1][a:a + L][:CD] < 0).all() and (m0["pdepth"][a:a + L][:CD] < 0).all()
        assert np.array_equal(m0["ids"][1][a:a + L], np.asarray(row["lane"])[s0:s0 + L]), "text lane frame s0 + q at q"
        a += L
    m1 = D.packed(m0, cfg)
    assert m1["segs"] == [len(m1["pid"])] and np.array_equal(m1["pid"], np.arange(len(m1["pid"])))
print(f"pose rows: Mimi and pose codes of frame s0 + q - {CD} at position q (same delay), lane frame at q; packed = one causal segment")
tm = D.TextDS("/data/home/amartyavarma/data/gemma-txtdup-v1", "train", [0], dict(cfg, txt_cap=7000))[0]
assert tm["segs"] == [len(tm["pid"])] and np.array_equal(tm["pid"], np.arange(len(tm["pid"])))
print(f"CD ALIGNMENT TEST OK (CD={CD})")
