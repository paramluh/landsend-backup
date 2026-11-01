#!/usr/bin/env python3
"""pose5 G0COND tests: compare JSON outputs of test_cfg.py / run_variant.py -> PASS / FAIL line + out/<name>.json.

  compare.py cfg  <orig.json> <copy_off.json> <copy_on.json> <out.json>
  compare.py var  <base.json> <other.json> <expect> <out.json>
    expect: same      every micro's stats / loss / every gradient / the decoders after their nanosteps bit-identical (bytes), same params
            g0_zero   = same on every shared tensor; the other has exactly inp.g0.weight more; its gradient: non-zero on the backbone-FULL
                      pose micro (every column whose feature is non-zero on some clip), None on the FROZEN pose micro and on speech / text
            g0_nz     pose micros' terms differ; speech / text micros bit-identical (stats, loss, every shared gradient, decoders)
            both_zero (merged copy) = same on every shared tensor; extra = inp.g0.weight + tower_p.s_emb.weight; g0 as g0_zero; s_emb grad
                      non-zero on both pose micros, None on speech / text"""
import json
import sys

G0, S2 = "inp.g0.weight", "tower_p.s_emb.weight"


def load(p):
    return json.load(open(p))


def cfg(a):
    o, c0, c1 = (load(x) for x in a[:3])
    res = {}
    res["flag_off_cfg_identical"] = o["CFG_json"] == c0["CFG_json"]
    res["flag_off_coef_resumekeys_nv_groups_identical"] = all(o[k] == c0[k] for k in ("COEF", "RESUME_KEYS", "NV", "SEQ", "GROUPS"))
    res["flag_off_params_identical"] = o["params"] == c0["params"]
    res["flag_off_data_cfg_identical"] = o["data_cfg"] == c0["data_cfg"]
    dk = sorted(set(c1["CFG"]) ^ set(o["CFG"]))
    same_rest = all(json.dumps(c1["CFG"][k], sort_keys=True) == json.dumps(o["CFG"][k], sort_keys=True) for k in o["CFG"])
    res["flag_on_cfg_diff_keys"] = dk
    res["flag_on_cfg_values"] = {k: c1["CFG"].get(k) for k in dk}
    res["flag_on_cfg_only_g0_keys_added"] = dk == ["g0cond", "g0cond_from_step"] and c1["CFG"]["g0cond"] == 1 and same_rest
    res["flag_on_data_cfg_identical"] = o["data_cfg"] == c1["data_cfg"]
    res["flag_on_resume_keys_identical"] = o["RESUME_KEYS"] == c1["RESUME_KEYS"]
    extra = [p for p in c1["params"] if p not in o["params"]]
    res["flag_on_params_extra"] = extra
    res["flag_on_params_only_g0_added"] = extra == [[G0, [5120, 8], "audio"]] and [p for p in c1["params"] if p[0] != G0] == o["params"]
    res["pose_input_embedding_group"] = [p[2] for p in o["params"] if p[0] == "inp.pose.weight"]
    res["g0_in_pose_input_embedding_group"] = res["pose_input_embedding_group"] == ["audio"] and extra[:1] and extra[0][2] == "audio"
    res["g0_in_backbone_groups_811"] = "audio" in ("trunk", "spec", "proj", "audio")
    ok = all(v for k, v in res.items() if isinstance(v, bool))
    res["PASS"] = ok
    json.dump(res, open(a[3], "w"), indent=1)
    print(("PASS" if ok else "FAIL"), "cfg:", json.dumps({k: v for k, v in res.items() if isinstance(v, bool)}))
    for k in ("flag_on_cfg_values", "flag_on_params_extra"):
        print("   ", k, res[k])


def var(a):
    A, B, expect, outp = load(a[0]), load(a[1]), a[2], a[3]
    res = {"base": f"{A['code']} g0 {A.get('g0', A.get('variant'))} s2p {A.get('s2p')} {A['model']}", "other": f"{B['code']} g0 {B['g0']} s2p {B['s2p']} {B['model']}",
           "expect": expect, "micros": {}}
    shared = [n for n in A["init_sha"] if n in B["init_sha"]]
    res["init_shared_identical"] = all(A["init_sha"][n] == B["init_sha"][n] for n in shared)
    res["params_set"] = {"base_only": sorted(set(A["init_sha"]) - set(B["init_sha"])), "other_only": sorted(set(B["init_sha"]) - set(A["init_sha"]))}
    ok = res["init_shared_identical"]
    want_extra = {"same": [], "g0_zero": [G0], "g0_nz": [G0], "both_zero": sorted([G0, S2])}[expect]
    ok &= res["params_set"] == {"base_only": [], "other_only": want_extra}
    if expect == "same":
        ok &= A["param_shapes_sha"] == B["param_shapes_sha"]
    for mk in A["micros"]:
        x, y = A["micros"][mk], B["micros"][mk]
        pose = mk.startswith("pose")
        r = {"v_identical": x["v"] == y["v"], "loss_identical": x["loss"] == y["loss"], "loss": [x["loss_float"], y["loss_float"]]}
        gs = [n for n in x["grad_sha"] if n not in (G0, S2)]
        diff = [n for n in gs if x["grad_sha"][n] != y["grad_sha"].get(n)]
        vdiff = [n for n in gs if x["grad_vsha"][n] != y["grad_vsha"].get(n)]
        r.update(grads_compared=len(gs), grads_set=sum(x["grad_sha"][n] is not None for n in gs), grads_byte_differ=diff[:6], n_grads_byte_differ=len(diff),
                 grads_value_differ=vdiff[:6], n_grads_value_differ=len(vdiff), dec_after_identical=x["dec_after_sha"] == y["dec_after_sha"],
                 seconds=[x["seconds"], y["seconds"]])
        g0r = y.get("g0")
        r["g0"] = g0r
        same_all = r["v_identical"] and r["loss_identical"] and not diff and r["dec_after_identical"]
        if expect in ("same", "g0_zero", "both_zero"):
            m_ok = same_all
        else:                                                      # g0_nz: only pose micros change
            m_ok = (not r["loss_identical"] and not r["v_identical"]) if pose else same_all
        if expect in ("g0_zero", "g0_nz", "both_zero"):
            if mk == "pose_full":
                cols = g0r.get("col_norms") or []
                m_ok &= bool(g0r.get("grad_norm")) and g0r["grad_norm"] > 0 and g0r.get("n_valid", 0) > 0 and len(cols) == 8 and all((c > 0) == fz for c, fz in zip(cols, g0r["f_col_nonzero"]))
                if "recomputed_rel_err" in g0r:
                    m_ok &= g0r["recomputed_rel_err"] < 2e-2
            else:
                m_ok &= g0r == {"grad": None}
        if expect == "both_zero":
            sx = y["grad_norm"].get(S2)
            r["s_emb_grad_norm"] = sx
            m_ok &= (sx is not None and sx > 0) if pose else (y["grad_sha"].get(S2) is None)
        r["PASS"] = bool(m_ok)
        ok &= m_ok
        res["micros"][mk] = r
    res["PASS"] = bool(ok)
    json.dump(res, open(outp, "w"), indent=1)
    print(("PASS" if ok else "FAIL"), f"var {expect}: {res['base']} vs {res['other']}: " +
          "; ".join(f"{k} {'ok' if v['PASS'] else 'FAIL'} (loss {v['loss'][0]:.6f}/{v['loss'][1]:.6f}, byte-diff grads {v['n_grads_byte_differ']}/{v['grads_compared']}"
                    + (f", g0 {v['g0'].get('grad_norm') if v['g0'] else None}" if v['g0'] is not None else "") + ")" for k, v in res["micros"].items()))


if __name__ == "__main__":
    {"cfg": cfg, "var": var}[sys.argv[1]](sys.argv[2:])
