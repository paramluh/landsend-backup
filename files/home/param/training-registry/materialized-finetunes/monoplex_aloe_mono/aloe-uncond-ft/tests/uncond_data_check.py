"""aloe-uncond-ft CPU check: for a sample of train + eval items, the A-empty micro vs 22091's micro (data_ppm.speech_micro + pose):
A = PPM one-sided form everywhere; B's prefix tokens / clip codes / persona == 22091's B-owned prefix positions; B lane / Mimi / pose /
every target == 22091's shifted by the prefix difference."""
import os, sys, json
os.environ["QQA_A_EMPTY"] = "1"; os.environ["QQA_POSE_B"] = "1"
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-uncond-ft/code")
import numpy as np
import data_aloe_pose as A
D = A.D
cfg = A.aloe_cfg()
sp, nrows = A.splits()
rng = np.random.default_rng(0)
items = [sp["train"][i] for i in rng.choice(len(sp["train"]), 120, replace=False)] + [sp["eval"][i] for i in rng.choice(len(sp["eval"]), 40, replace=False)]
st = A.Stores()
n_clipA = n_persA = n_clipB = n_persB = n_cut = 0
dP = []
for it in items:
    k, r, sw = it
    node, rs, sib, ps, pz, b_ok = st.open(k)
    row = rs[r]
    try:
        pers = ps.get(r, row["row_key"])
    except (FileNotFoundError, OSError):
        pers = None
    m0 = D.speech_micro(row, r, sib, pers, cfg, 0, swap=sw)
    m1 = A.speech_micro_a_empty(row, r, sib, pers, cfg, 0, swap=sw)
    P0, P1 = int(m0["P"]), int(m1["P"])
    assert (m1["ids"][0] == D.PAD_ID).all() and (m1["depth"][0] == -1).all() and (m1["present"][0] == 0).all()
    own = m0["present"][1, :P0] > 0
    assert own.sum() == P1, (it, own.sum(), P1)
    assert np.array_equal(m0["ids"][1, :P0][own], m1["ids"][1, :P1]) and np.array_equal(m0["depth"][1, :P0][own], m1["depth"][1, :P1])
    assert (m1["present"][1, :P1] == 1).all()
    assert np.array_equal(m0["ids"][1, P0:], m1["ids"][1, P1:]) and np.array_equal(m0["depth"][1, P0:], m1["depth"][1, P1:])
    assert np.array_equal(m0["present"][1, P0:], m1["present"][1, P1:])
    assert np.array_equal(m0["sel"] - P0, m1["sel"] - P1) and np.array_equal(m0["light"] - P0, m1["light"] - P1)
    for key in ("y", "word", "pad", "eob", "nw"):
        assert np.array_equal(m0[key], m1[key]), key
    assert np.array_equal(m0["cb0"][P0:], m1["cb0"][P1:]) and (m1["cb0"][:P1] < 0).all()
    # A in 22091's conversation part was real content (sanity: we did remove something)
    assert (m0["present"][0, P0:] == 1).all()
    # pose arrays on each micro
    m0["swap"], m1["swap"] = sw, sw
    bside = "user" if sw else "assistant"
    p0 = A.pose_arrays(m0, row, pz[bside][r], cfg); p1 = A.pose_arrays(m1, row, pz[bside][r], cfg)
    assert np.array_equal(p0[0][P0:], p1[0][P1:]) and (p1[0][:P1] < 0).all() and np.array_equal(p0[1][P0:], p1[1][P1:])
    assert np.array_equal(p0[2] - P0, p1[2] - P1)
    pv0, pv1 = json.loads(m0["prov"]), json.loads(m1["prov"])
    n_clipA += sum(c["track"] == "A" for c in pv0["clips"]); n_clipB += sum(c["track"] == "B" for c in pv0["clips"])
    n_persA += sum(c["track"] == "A" for c in pv0["persona"]); n_persB += sum(c["track"] == "B" for c in pv0["persona"])
    n_cut += int(pv0["persona_cut"] > 0)
    dP.append(P0 - P1)
    # full conv path (Stores.conv with A_EMPTY) equals m1
    mc = st.conv(it, cfg, 0)
    assert np.array_equal(mc["ids"], m1["ids"]) and np.array_equal(mc["present"], m1["present"]) and np.array_equal(mc["pdepth"], p1[0])
print(f"OK {len(items)} items ({len(sp['train'])} train / {len(sp['eval'])} eval items total): 22091 prefixes had A clips {n_clipA}, B clips {n_clipB}, "
      f"A persona {n_persA}, B persona {n_persB}, persona crop on {n_cut}; prefix shrink P0-P1 mean {np.mean(dP):.0f} max {max(dP)} min {min(dP)}")
