#!/usr/bin/env python3
"""pose5 G0COND + POSE-S2P COMBINED comparisons (merged copy), generalising compare.py (which stays the G0COND-only record).

  compare2.py cfg <out.json> <orig.json> <name=cfg.json> ...     names: off, g0, cb0, full, g0cb0, g0full (test_cfg.py outputs)
      off == ORIG byte for byte (CFG json, COEF, RESUME_KEYS, NV, GA, groups, params, data cfg); every other variant = ORIG + exactly its
      own cfg keys (g0cond + g0cond_from_step; s2p + s2p_from_step) and exactly its own params (inp.g0.weight [5120, 8] 'audio';
      tower_p.s_emb.weight [65536 | 2048, 2560] 'tower_p'), CFG['data'] and RESUME_KEYS unchanged.
  compare2.py var <base.json> <other.json> <expect> <out.json>
      base = ORIG flag off (run_variant.py output; A's format accepted). expect = '+'-joined new keys present in other, with zero / nz:
        z:g0, z:cb0, z:full, z:g0+cb0, z:g0+full  -> every micro's stats / loss / every shared gradient / decoders bit-identical; inp.g0 grad
            non-zero on the backbone-FULL pose micro only (None on the frozen pose micro and on speech / text); s_emb grad non-zero on both
            pose micros, None on speech / text
        nz:<keys>  -> pose micros change; speech / text micros bit-identical (stats, loss, shared grads, decoders)"""
import json
import sys

G0, S2 = "inp.g0.weight", "tower_p.s_emb.weight"
WANT = {"g0": ({"g0cond": 1}, [[G0, [5120, 8], "g0cond"]]), "cb0": ({"s2p": "cb0"}, [[S2, [2048, 2560], "s2p"]]),
        "full": ({"s2p": "full"}, [[S2, [65536, 2560], "s2p"]])}
NEWG = {"g0": "g0cond", "cb0": "s2p", "full": "s2p"}                # own optimizer groups (audit S7), JOIN: g0cond <- audio, s2p <- tower_p
JOIN = {"g0cond": "audio", "s2p": "tower_p"}


def load(p):
    return json.load(open(p))


def cfg(a):
    outp, o = a[0], load(a[1])
    res, ok = {}, True
    for arg in a[2:]:
        name, path = arg.split("=", 1)
        x = load(path)
        r = {}
        r["GROUPS_code_tuple_adds_exactly_new_two"] = x["GROUPS"] == o["GROUPS"] + ["g0cond", "s2p"]
        if name == "off":
            r["cfg_identical"] = o["CFG_json"] == x["CFG_json"]
            r["coef_resumekeys_nv_seq_identical"] = all(o[k] == x[k] for k in ("COEF", "RESUME_KEYS", "NV", "SEQ"))
            r["optimizer_groups_identical"] = x["opt_groups"] == o["opt_groups"]
            r["params_identical"] = o["params"] == x["params"]
            r["data_cfg_identical"] = o["data_cfg"] == x["data_cfg"]
        else:
            parts = ["g0"] * name.startswith("g0") + [p for p in ("cb0", "full") if name.endswith(p)]
            keys, params = {}, []
            for p in parts:
                keys.update(WANT[p][0]); params += WANT[p][1]
            want_keys = sorted(sum([[k, k + "_from_step"] for k in keys], []))
            dk = sorted(set(x["CFG"]) ^ set(o["CFG"]))
            r["cfg_diff_keys"] = dk
            ng = sorted({NEWG[p] for p in parts})
            r["cfg_only_own_keys_added"] = dk == want_keys and all(x["CFG"][k] == v for k, v in keys.items()) and \
                all(x["CFG"][k + "_from_step"] == 0 for k in keys) and \
                all(json.dumps(x["CFG"][k], sort_keys=True) == json.dumps(o["CFG"][k], sort_keys=True) for k in o["CFG"] if k not in ("lr", "clip"))
            r["lr_clip_only_new_groups_added_with_join_values"] = all(
                sorted(set(x["CFG"][s]) - set(o["CFG"][s])) == ng and all(x["CFG"][s][g] == o["CFG"][s][g] for g in o["CFG"][s]) and
                all(x["CFG"][s][g] == o["CFG"][s][JOIN[g]] for g in ng) for s in ("lr", "clip"))
            r["lr_new"] = {g: [x["CFG"]["lr"][g], x["CFG"]["clip"][g]] for g in ng}
            r["optimizer_groups_old_plus_new"] = x["opt_groups"] == [g for g in x["GROUPS"] if g in o["opt_groups"] or g in ng]
            r["warmup_groups_identical"] = x["CFG"]["warmup_groups"] == o["CFG"]["warmup_groups"]
            extra = [p for p in x["params"] if p not in o["params"]]
            r["params_extra"] = extra
            r["params_only_own_added"] = sorted(map(str, extra)) == sorted(map(str, params)) and \
                [p for p in x["params"] if p[0] not in (G0, S2)] == o["params"]
            r["data_cfg_identical"] = o["data_cfg"] == x["data_cfg"]
            r["resume_keys_identical"] = o["RESUME_KEYS"] == x["RESUME_KEYS"]
        r["PASS"] = all(v for v in r.values() if isinstance(v, bool))
        ok &= r["PASS"]
        res[name] = r
    res["PASS"] = bool(ok)
    json.dump(res, open(outp, "w"), indent=1)
    print(("PASS" if ok else "FAIL"), "cfg combos:", {k: v["PASS"] for k, v in res.items() if isinstance(v, dict)})
    for k, v in res.items():
        if isinstance(v, dict) and v.get("params_extra") is not None:
            print("   ", k, v.get("cfg_diff_keys"), v.get("params_extra"))


def var(a):
    A, B, expect, outp = load(a[0]), load(a[1]), a[2], a[3]
    mode, keys = expect.split(":")
    keys = keys.split("+")
    has_g0 = "g0" in keys
    has_s2 = any(k in ("cb0", "full") for k in keys)
    res = {"base": f"{A['code']} {A.get('variant', 'g0 ' + str(A.get('g0')) + ' s2p ' + str(A.get('s2p')))} {A['model']}",
           "other": f"{B['code']} g0 {B.get('g0')} s2p {B.get('s2p')} {B['model']}", "expect": expect, "micros": {}}
    shared = [n for n in A["init_sha"] if n in B["init_sha"]]
    res["init_shared_identical"] = all(A["init_sha"][n] == B["init_sha"][n] for n in shared)
    res["params_set"] = {"base_only": sorted(set(A["init_sha"]) - set(B["init_sha"])), "other_only": sorted(set(B["init_sha"]) - set(A["init_sha"]))}
    ok = res["init_shared_identical"] and res["params_set"] == {"base_only": [], "other_only": sorted([G0] * has_g0 + [S2] * has_s2)}
    for mk in A["micros"]:
        x, y = A["micros"][mk], B["micros"][mk]
        pose = mk.startswith("pose")
        gs = [n for n in x["grad_sha"] if n not in (G0, S2)]
        diff = [n for n in gs if x["grad_sha"][n] != y["grad_sha"].get(n)]
        r = {"v_identical": x["v"] == y["v"], "loss_identical": x["loss"] == y["loss"], "loss": [x["loss_float"], y["loss_float"]],
             "grads_compared": len(gs), "grads_set": sum(x["grad_sha"][n] is not None for n in gs), "n_grads_byte_differ": len(diff),
             "grads_byte_differ": diff[:6], "dec_after_identical": x["dec_after_sha"] == y["dec_after_sha"],
             "g0_grad_norm": y["grad_norm"].get(G0), "s_emb_grad_norm": y["grad_norm"].get(S2)}
        same_all = r["v_identical"] and r["loss_identical"] and not diff and r["dec_after_identical"]
        if "group_norm" in x and "group_norm" in y:                # every EXISTING group's grad norm + clip factor bit-identical (own-group design)
            og = sorted(x["group_norm"])
            r["existing_group_norms_identical"] = all(x["group_norm"][g] == y["group_norm"].get(g) for g in og) and \
                all(x["group_clip_factor"].get(g) == y["group_clip_factor"].get(g) for g in og)
            r["new_groups"] = {g: y["group_norm"][g] for g in y["group_norm"] if g not in x["group_norm"]}
            if mode == "z" or not pose:
                same_all &= r["existing_group_norms_identical"]
        m_ok = same_all if (mode == "z" or not pose) else (not r["loss_identical"] and not r["v_identical"])
        if has_g0:
            g = r["g0_grad_norm"]
            m_ok &= (g is not None and g > 0) if mk == "pose_full" else (y["grad_sha"].get(G0) is None)
        if has_s2:
            g = r["s_emb_grad_norm"]
            m_ok &= (g is not None and g > 0) if pose else (y["grad_sha"].get(S2) is None)
        r["PASS"] = bool(m_ok)
        ok &= m_ok
        res["micros"][mk] = r
    res["PASS"] = bool(ok)
    json.dump(res, open(outp, "w"), indent=1)
    print(("PASS" if ok else "FAIL"), f"var {expect}: {res['base']} vs {res['other']}: " +
          "; ".join(f"{k} {'ok' if v['PASS'] else 'FAIL'} (loss {v['loss'][0]:.6f}/{v['loss'][1]:.6f}, byte-diff grads {v['n_grads_byte_differ']}/{v['grads_compared']}, "
                    f"g0 {v['g0_grad_norm']}, s_emb {v['s_emb_grad_norm']})" for k, v in res["micros"].items()))


if __name__ == "__main__":
    {"cfg": cfg, "var": var}[sys.argv[1]](sys.argv[2:])
