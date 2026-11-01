#!/usr/bin/env python3
"""pose6 B1 test stage A: REAL Aloe micros built by a code dir's own data path (Stores.conv -> as_micro / pack_micro, the trainer's
StepDS / eval_micros_v2 path), with the mono8020 env + this process's PPM_G0COND / PPM_S2P.

  --select (pose6 code, PPM_G0COND=1): pick one short item per prompt type (EMPTY block, CLIP only, PERSONA only, BOTH) from the eval split
           (train split if eval lacks a type), write SCR/items.json; record the SLOT facts per type (D1: what sits at P - 1, what at P,
           the first speech / pose label positions); check g0f == the sidecar's stored f BYTE-EXACTLY on these + 24 more random items;
           the sidecar's coverage of EVERY train + eval item (splits_v2, = the plan's items).
  --tag T  build the SAME items (items.json) -> SCR/micros_T.pkl: {'singles': [as_micro(item)], 'packed': pack_micro(all 4), 'packed2':
           pack_micro(items 2..3)} (singles / packed = the trainer's exact micro objects)."""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h6 as H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--select", action="store_true")
ap.add_argument("--maxn", type=int, default=1200)
a = ap.parse_args()
H.use_code(a.code)
import data_aloe_pose as A  # noqa: E402
import train_aloe_pose as T  # noqa: E402   (module level only: CFG; no dist)

assert os.path.dirname(os.path.abspath(A.__file__)) == a.code and os.path.dirname(os.path.abspath(T.__file__)) == a.code
os.makedirs(H.SCR, exist_ok=True)
cfg = T.CFG["data"]
st = A.Stores()
G0 = bool(getattr(A, "G0", False))
H.log(f"code {a.code} tag {a.tag}: G0 {G0} S2P {getattr(T, 'S2P', None)}; data cfg {cfg}")
TYPES = ("empty", "clip", "persona", "both")


def ptype(m):
    pv = json.loads(m["prov"])
    c, p = len(pv["clips"]) > 0, len(pv["persona"]) > 0
    return {(False, False): "empty", (True, False): "clip", (False, True): "persona", (True, True): "both"}[(c, p)]


if a.select:
    assert G0, "--select runs with PPM_G0COND=1"
    sp = A.splits_v2()
    pick = {}
    for split in ("eval", "train"):                               # every item's prompt type from the micro's own prefix builder (=
        if len(pick) == 4:                                         # prompt_stats_mono: podcast_prefix_a_empty), then the shortest per type
            break
        best = {}
        for it in sorted(sp[split], key=lambda x: (x[3], x[4]))[: (None if split == "eval" else 6000)]:
            if it[3] < 40:
                continue
            node, rows, sib, ps, pz, b_ok = st.open(int(it[0]))
            row = rows[int(it[1])]
            try:
                pers = ps.get(int(it[1]), row["row_key"])
            except (FileNotFoundError, OSError):
                pers = None
            _, pv = A.podcast_prefix_a_empty(row, int(it[2]), sib, pers, cfg)
            t = {(False, False): "empty", (True, False): "clip", (False, True): "persona", (True, True): "both"}[(len(pv["clips"]) > 0, len(pv["persona"]) > 0)]
            if t not in pick and t not in best:
                best[t] = {"split": split, "item": list(it[:3]), "n_frames": int(it[3]), "row_key": it[4]}
                H.log("picked", t, best[t])
        for t, v in best.items():
            pick.setdefault(t, v)
    assert sorted(pick) == sorted(TYPES), ("prompt types found", sorted(pick))
    items = [pick[t] for t in TYPES]
    json.dump(items, open(f"{H.SCR}/items.json", "w"), indent=1)
    # ---- SLOT facts per prompt type (D1)
    facts = {}
    for t, x in zip(TYPES, items):
        m = st.conv(x["item"], cfg, 0)
        P = int(m["P"])
        ids, dep, pres, pd, pcb0 = (np.asarray(m[k]) for k in ("ids", "depth", "present", "pdepth", "pcb0"))
        hs = np.flatnonzero(dep[1, :, 0] >= 0)
        facts[t] = {"item": x, "P": P, "T": int(len(m["pid"])), "lane_B[P-3..P+1]": ids[1, P - 3:P + 2].tolist(),
                    "lane_A[P-1]": int(ids[0, P - 1]), "present_B[P-1]": float(pres[1, P - 1]), "present_A[P-1]": float(pres[0, P - 1]),
                    "codes_at_P-1 (any channel)": bool((dep[:, P - 1] >= 0).any()), "pose_codes_at_P-1": bool((pd[P - 1] >= 0).any()),
                    "text_target_at_P-1": bool((np.asarray(m["sel"]) == P - 1).any()), "first_text_target": int(np.asarray(m["sel"]).min()),
                    "speech_codes_in_block": int((hs < P).sum()), "first_conv_speech_code": int(hs[hs >= P].min()),
                    "first_pose_label": int(np.flatnonzero(pcb0 >= 0).min()), "SLOT": P - 1,
                    "first_pose_label - SLOT": int(np.flatnonzero(pcb0 >= 0).min()) - (P - 1), "g0f": m["g0f"].tolist(),
                    "prov": json.loads(m["prov"])}
        H.log("SLOT", t, {k: v for k, v in facts[t].items() if k != "prov"})
    # ---- g0f == sidecar f byte-exactly on real micros (B2's request): the 4 + 24 random items (eval + train)
    import pyarrow.parquet as pq
    tab = {(int(r["store"]), int(r["row"]), int(r["swap"])): r for r in pq.read_table(H.SIDECAR, columns=["store", "row", "swap", "f", "item_key"]).to_pylist()}
    rng = np.random.default_rng(20261027)
    extra = [tuple(sp["eval"][i][:3]) for i in rng.choice(len(sp["eval"]), 12, replace=False)] + \
            [tuple(sp["train"][i][:3]) for i in rng.choice(len(sp["train"]), 12, replace=False)]
    chk = []
    for it in [tuple(x["item"]) for x in items] + extra:
        m = st.conv(list(it), cfg, 0)
        mic = A.pack_micro([m])
        f_s = np.asarray(tab[tuple(int(v) for v in it)]["f"], np.float32)
        chk.append({"item": list(map(int, it)), "row_key": m["row_key"], "P": int(m["P"]), "g0pos": mic["g0pos"].tolist(),
                    "g0f_bytes_equal_sidecar": bool(np.asarray(m["g0f"]).dtype == np.float32 and np.asarray(m["g0f"]).tobytes() == f_s.tobytes()
                                                    and mic["g0f"][0].tobytes() == f_s.tobytes()),
                    "g0pos_is_P-1": bool(mic["g0pos"].tolist() == [int(m["P"]) - 1])})
    ok_f = all(c["g0f_bytes_equal_sidecar"] and c["g0pos_is_P-1"] for c in chk)
    H.log(f"g0f == sidecar f byte-exact on {len(chk)} real micros: {ok_f}")
    # ---- coverage of every item (= the plan's items; data_aloe_pose.g0_coverage asserts each)
    n_tr, inv_tr = A.g0_coverage(sp["train"])
    n_ev, inv_ev = A.g0_coverage(sp["eval"])
    H.log(f"coverage: train {n_tr} (f = 0: {inv_tr}), eval {n_ev} (f = 0: {inv_ev})")
    # ---- a missing item fails loud
    try:
        A.g0_item((99, 0, 0), "nonexistent")
        miss_loud = False
    except AssertionError as e:
        miss_loud = "missing from the frame-0 sidecar" in str(e)
    json.dump({"slot_facts": facts, "g0f_checks": chk, "g0f_all_byte_equal": ok_f, "coverage": {"train": [n_tr, inv_tr], "eval": [n_ev, inv_ev]},
               "missing_item_fails_loud": miss_loud, "stats": dict(A.G0_STATS)}, open(f"{H.OUTD}/stageA_select.json", "w"), indent=1, default=str)
    assert ok_f and miss_loud and inv_tr == 0 and inv_ev == 0, "stage A select checks"

items = json.load(open(f"{H.SCR}/items.json"))
ms = [st.conv(x["item"], cfg, 0) for x in items]
Tn = [len(m["pid"]) for m in ms]
order = sorted(range(len(ms)), key=lambda i: Tn[i])               # packed = the most items (shortest first) that fit A.BUDGET, packed2 = the rest
k = max(j for j in range(1, len(ms) + 1) if sum(Tn[i] for i in order[:j]) <= A.BUDGET)
g1, g2 = sorted(order[:k]), sorted(order[k:]) or [order[-1]]
assert sum(Tn[i] for i in g2) <= A.BUDGET, ("second pack over the budget", Tn)
out = {"singles": [A.as_micro(m) for m in ms], "packed": A.pack_micro([ms[i] for i in g1]), "packed2": A.pack_micro([ms[i] for i in g2]),
       "P": [int(m["P"]) for m in ms], "T": Tn, "groups": [g1, g2]}
if G0:
    for key, g in (("packed", g1), ("packed2", g2)):
        o = np.concatenate([[0], np.cumsum([Tn[i] for i in g])[:-1]])
        assert out[key]["g0pos"].tolist() == [int(x + int(ms[i]["P"]) - 1) for x, i in zip(o, g)], "packed g0pos = offset + P - 1"
        assert np.array_equal(out[key]["g0f"], np.stack([ms[i]["g0f"] for i in g]))
pickle.dump(out, open(f"{H.SCR}/micros_{a.tag}.pkl", "wb"))
H.log(f"micros_{a.tag}: P {out['P']} T {out['T']}; packed positions {len(out['packed']['pid'])}; keys {sorted(out['packed'])}")
