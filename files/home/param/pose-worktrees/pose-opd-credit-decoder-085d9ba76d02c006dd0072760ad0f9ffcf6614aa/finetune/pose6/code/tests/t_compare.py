#!/usr/bin/env python3
"""pose6 B1: compare stage A micros + stage B variants -> out/stageAB_compare.json (+ PASS).
  A  micros_off (pose6 data code, flags off) == micros_mono (aloe-pose-mono-8020's data code), every key / array byte-identical;
     micros_on minus {g0pos, g0f} == micros_off (G0COND adds only those two keys).
  B  (1) mono (mono8020 trainer + model) vs off (pose6, flags off): resolved CFG, forward terms, stats / loss, EVERY gradient, decoders after
         their nanosteps, group clip norms, every parameter after the AdamW step: all identical.
     (5) off vs on0 (pose6, PPM_S2P=cb0 + PPM_G0COND=1, new params = 0): the same, on every pre-existing tensor; the new params get non-zero
         gradients and non-zero values after the step (own groups: no other update changes).
     (4) on0: s_emb gradient rows only on cb0 codes present; inp.g0's output gradient == the trunk-input gradient at the SLOT positions
         (bitwise) and dW = sum_k dy_k (x) f_k.
     nz: on with non-zero new params: forward terms differ from off (both mods act)."""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h6 as H  # noqa: E402

OUT, ok_all = {}, True


def same(x, y):
    if isinstance(x, dict):
        return isinstance(y, dict) and sorted(x) == sorted(y) and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)) and not (len(x) and isinstance(x[0], (int, float, np.integer, np.floating))):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if isinstance(x, (np.ndarray, list, tuple)) or isinstance(y, np.ndarray):
        xa, ya = np.asarray(x), np.asarray(y)
        return xa.dtype == ya.dtype and xa.shape == ya.shape and xa.tobytes() == ya.tobytes()
    return x == y


mm = {t: pickle.load(open(f"{H.SCR}/micros_{t}.pkl", "rb")) for t in ("mono", "off", "on")}
a1 = same(mm["off"], mm["mono"])
strip = lambda d: {k: v for k, v in d.items() if k not in ("g0pos", "g0f")}  # noqa: E731
a2 = all(same(strip(mm["on"][k]), mm["off"][k]) for k in ("packed", "packed2")) and \
    all(same(strip(x), y) for x, y in zip(mm["on"]["singles"], mm["off"]["singles"]))
a3 = all("g0pos" in mm["on"][k] for k in ("packed", "packed2")) and not any("g0pos" in mm["off"][k] for k in ("packed", "packed2"))
OUT["A"] = {"micros_off_eq_mono8020": a1, "micros_on_minus_g0_eq_off": a2, "g0_keys_only_when_on": a3,
            "packed_g0pos": mm["on"]["packed"]["g0pos"].tolist(), "packed2_g0pos": mm["on"]["packed2"]["g0pos"].tolist(),
            "pack_groups": mm["on"]["groups"], "P": mm["on"]["P"], "T": mm["on"]["T"]}
ok_all &= a1 and a2 and a3
H.log("A", OUT["A"])

V = {v: json.load(open(f"{H.OUTD}/stageB_{v}.json")) for v in ("mono", "off", "on0", "onnz", "on0bboff")}


def flat(d, pre=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flat(v, f"{pre}{k}."))
        else:
            o[f"{pre}{k}"] = v
    return o


def cmp(x, y, new_ok=()):
    """identical on every shared key (+ the new params may exist only in y)."""
    r = {}
    for part in ("fwd", "grads", "dec_after_nano", "after_step", "gn"):
        X, Y = x[part], y[part]
        extra = sorted(set(Y) - set(X))
        diff = sorted(k for k in set(X) & set(Y) if X[k] != Y[k])
        r[part] = {"identical_shared": not diff and not (set(X) - set(Y)), "differ": diff[:8], "only_in_2nd": extra}
    r["stats"] = x["stats"] == y["stats"]
    r["loss"] = x["loss"] == y["loss"]
    r["grad_none_same"] = x["grad_none"] == y["grad_none"]
    fx, fy = flat(x["cfg"]), flat(y["cfg"])
    r["cfg_diff"] = sorted(k for k in set(fx) | set(fy) if fx.get(k, "<absent>") != fy.get(k, "<absent>"))
    return r


r1 = cmp(V["mono"], V["off"])
ok1 = all(r1[p]["identical_shared"] and not r1[p]["only_in_2nd"] for p in ("fwd", "grads", "dec_after_nano", "after_step", "gn")) and \
    r1["stats"] and r1["loss"] and r1["grad_none_same"] and r1["cfg_diff"] == []
OUT["(1) mono8020 vs pose6 flags off"] = dict(r1, PASS=ok1)
r5 = cmp(V["off"], V["on0"])
NEWG = ["g0cond", "s2p"]
ok5 = all(r5[p]["identical_shared"] for p in ("fwd", "grads", "dec_after_nano", "after_step", "gn")) and r5["stats"] and r5["loss"] and \
    sorted(r5["grads"]["only_in_2nd"]) == sorted(H.NEW) and sorted(r5["after_step"]["only_in_2nd"]) == sorted(H.NEW) and \
    sorted(r5["gn"]["only_in_2nd"]) == NEWG and r5["fwd"]["only_in_2nd"] == [] and \
    all(V["on0"][f"grad_norm[{n}]"] > 0 and V["on0"][f"after_step_absmax[{n}]"] > 0 for n in H.NEW)
want_cfg = sorted(["clip.g0cond", "clip.s2p", "g0_sidecar", "g0cond", "lr.g0cond", "lr.s2p", "s2p"])
ok5c = r5["cfg_diff"] == want_cfg and V["on0"]["cfg"]["lr"]["s2p"] == V["on0"]["cfg"]["lr"]["tower_p"] and \
    V["on0"]["cfg"]["lr"]["g0cond"] == V["on0"]["cfg"]["lr"]["audio"] and V["on0"]["cfg"]["clip"]["s2p"] == V["on0"]["cfg"]["clip"]["tower_p"] and \
    V["on0"]["cfg"]["clip"]["g0cond"] == V["on0"]["cfg"]["clip"]["audio"]
OUT["(5) pose6 off vs on (zero new params)"] = dict(r5, cfg_diff_expected=want_cfg, cfg_join_ok=ok5c, groups_on=V["on0"]["groups"],
                                                    new_grad_norms={n: V["on0"][f"grad_norm[{n}]"] for n in H.NEW},
                                                    new_after_step_absmax={n: V["on0"][f"after_step_absmax[{n}]"] for n in H.NEW},
                                                    gn_new={k: float.fromhex(V["on0"]["gn"][k]) for k in NEWG}, PASS=ok5 and ok5c)
gp = V["on0"]["g0_grad_path"]
ok4 = V["on0"]["s_emb_rows_eq_codes_present"] and all(r["y_grad_eq_trunk_input_grad_at_SLOT"] for r in gp["per_round"]) and gp["rel"] < 2e-2
OUT["(4) gradient structure (on0)"] = {"s_emb_rows_subset_of_cb0_codes_present": V["on0"]["s_emb_rows_eq_codes_present"],
                                       "s_emb_rows_nonzero_vs_codes": V["on0"]["s_emb_rows_nonzero_vs_codes"], "g0_grad_path": gp, "PASS": ok4}
rn = cmp(V["off"], V["onnz"])
oknz = (not rn["fwd"]["identical_shared"]) and set(rn["fwd"]["differ"]) >= {"hv_p", "pcb0"}
OUT["nz: off vs on (non-zero new params)"] = {"fwd_differ": rn["fwd"]["differ"], "fwd_val_off": V["off"]["fwd_val"], "fwd_val_nz": V["onnz"]["fwd_val"],
                                              "PASS": oknz}
L = V["on0bboff"]
DB = ["decp_bb.bias", "decp_bb.weight"]
rl = cmp(V["on0"], L)
okl = (not L["zb_p_computed"]) and V["on0"]["zb_p_computed"] and all(n in L["grad_none"] for n in DB) and \
    all(L["after_step"][n] == L["before"][n] for n in DB) and "decp_bb" not in [g for g, _ in L["groups"]] and \
    "decp_bb" in [g for g, _ in V["on0"]["groups"]] and rl["cfg_diff"] == ["decp_bb_off"] and \
    all(L["before"][n] == V["on0"]["before"][n] for n in L["before"])
OUT["LINK OFF: on vs on + QQA_DECP_BB_OFF=1"] = {"zb_p_computed_off": L["zb_p_computed"], "decp_bb_grad_none": [n in L["grad_none"] for n in DB],
                                                 "decp_bb_unchanged_after_step": [L["after_step"][n] == L["before"][n] for n in DB],
                                                 "groups": L["groups"], "cfg_diff_vs_on": rl["cfg_diff"], "fwd_differ_vs_on": rl["fwd"]["differ"],
                                                 "loss_on": float.fromhex(V["on0"]["loss"]), "loss_linkoff": float.fromhex(L["loss"]), "PASS": okl}
# exports (A1's request): train_cfg.json (DCP) + node-local meta.json side; unset = mono8020's bytes
E_ = {v: V[v]["export"] for v in V}
eo = E_["off"]["train_cfg"] == E_["mono"]["train_cfg"] and E_["off"]["meta_wo_time"] == E_["mono"]["meta_wo_time"] and "side" not in E_["off"]["meta_keys"]
el = E_["on0bboff"]
okx = eo and el["train_cfg"]["decp_bb_alpha"] == 0.0 and el["train_cfg"]["decp_bb_off"] == 1 and el["side"]["decp_bb_alpha"] == 0.0 and \
    el["side"]["decp_bb_off"] == 1 and el["side"]["s2p"] == "cb0" and el["side"]["g0cond"] == 1 and el["train_cfg"]["s2p"] == "cb0" and \
    el["train_cfg"]["g0cond"] == 1 and E_["on0"]["train_cfg"]["decp_bb_alpha"] == 1.0 and "decp_bb_off" not in E_["on0"]["train_cfg"]
OUT["exports"] = {"off_train_cfg_json_bytes_eq_mono8020": E_["off"]["train_cfg_bytes_eq"] if "train_cfg_bytes_eq" in E_["off"] else None,
                  "off_eq_mono8020 (train_cfg.json content, meta.json minus save_time)": eo,
                  "linkoff_train_cfg_side": {k: el["train_cfg"].get(k) for k in ("step", "decp_bb", "decp_bb_alpha", "decp_bb_off", "s2p", "g0cond")},
                  "linkoff_meta_side": el["side"], "on_train_cfg_side": {k: E_["on0"]["train_cfg"].get(k) for k in ("decp_bb_alpha", "decp_bb_off", "s2p", "g0cond")},
                  "PASS": okx}
ok_all &= ok1 and ok5 and ok5c and ok4 and oknz and okl and okx
OUT["PASS"] = bool(ok_all)
json.dump(OUT, open(f"{H.OUTD}/stageAB_compare.json", "w"), indent=1, default=str)
for k, v in OUT.items():
    H.log(k, v if not isinstance(v, dict) else {x: y for x, y in v.items() if x not in ("grads",)})
H.log("RESULT stages A+B", "PASS" if ok_all else "FAIL")
