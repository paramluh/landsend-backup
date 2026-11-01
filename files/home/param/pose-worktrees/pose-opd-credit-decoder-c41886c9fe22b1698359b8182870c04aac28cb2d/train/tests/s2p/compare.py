#!/usr/bin/env python3
"""pose5 s2p tests: compare the JSON outputs of test_cfg.py / run_variant.py -> PASS / FAIL lines + out/<name>.json.

  compare.py cfg  <orig.json> <copy_off.json> <copy_full.json> <copy_cb0.json> <out.json>
  compare.py var  <base.json> <other.json> <expect> <out.json>
    expect: same      every micro's stats / loss / every gradient / the decoders after their nanosteps bit-identical (bytes), same params
            s2p_zero  = same on every shared tensor + s_emb exists, grads non-zero on pose micros (rows = exactly the codes present),
                      NO grad on speech / text micros
            s2p_nz    pose micros' pose terms differ; speech / text micros bit-identical (s2p touches pose rows only)"""
import json
import sys


def load(p):
    return json.load(open(p))


def cfg(a):
    o, c0, cf, cc = (load(x) for x in a[:4])
    res = {}
    res["flag_off_cfg_identical"] = o["CFG_json"] == c0["CFG_json"]
    res["flag_off_coef_resumekeys_nv_groups_identical"] = all(o[k] == c0[k] for k in ("COEF", "RESUME_KEYS", "NV", "SEQ", "GROUPS"))
    res["flag_off_params_identical"] = o["params"] == c0["params"]
    for nm, x, mode, rows in (("full", cf, "full", 65536), ("cb0", cc, "cb0", 2048)):
        dk = sorted(set(x["CFG"]) ^ set(o["CFG"]))
        same_rest = all(json.dumps(x["CFG"][k], sort_keys=True) == json.dumps(o["CFG"][k], sort_keys=True) for k in o["CFG"])
        res[f"flag_{nm}_cfg_diff_keys"] = dk
        res[f"flag_{nm}_cfg_values"] = {k: x["CFG"].get(k) for k in dk}
        res[f"flag_{nm}_cfg_only_s2p_keys_added"] = dk == ["s2p", "s2p_from_step"] and x["CFG"]["s2p"] == mode and same_rest
        extra = [p for p in x["params"] if p not in o["params"]]
        res[f"flag_{nm}_params_extra"] = extra
        res[f"flag_{nm}_params_only_s_emb_added"] = (extra == [["tower_p.s_emb.weight", [rows, 2560], "tower_p"]]
                                                     and [p for p in x["params"] if p[0] != "tower_p.s_emb.weight"] == o["params"])
    ok = all(v for k, v in res.items() if isinstance(v, bool))
    res["PASS"] = ok
    json.dump(res, open(a[4], "w"), indent=1)
    print(("PASS" if ok else "FAIL"), "cfg:", json.dumps({k: v for k, v in res.items() if isinstance(v, bool)}))
    for k in ("flag_full_cfg_values", "flag_cb0_cfg_values", "flag_full_params_extra", "flag_cb0_params_extra"):
        print("   ", k, res[k])


def var(a):
    A, B, expect, outp = load(a[0]), load(a[1]), a[2], a[3]
    S = "tower_p.s_emb.weight"
    res = {"base": f"{A['code']} {A['variant']} {A['model']}", "other": f"{B['code']} {B['variant']} {B['model']}", "expect": expect, "micros": {}}
    shared = [n for n in A["init_sha"] if n in B["init_sha"]]
    res["init_shared_identical"] = all(A["init_sha"][n] == B["init_sha"][n] for n in shared)
    res["params_set"] = {"base_only": sorted(set(A["init_sha"]) - set(B["init_sha"])), "other_only": sorted(set(B["init_sha"]) - set(A["init_sha"]))}
    ok = res["init_shared_identical"]
    if expect == "same":
        ok &= A["param_shapes_sha"] == B["param_shapes_sha"] and not res["params_set"]["base_only"] and not res["params_set"]["other_only"]
    else:
        ok &= res["params_set"] == {"base_only": [], "other_only": [S]}
    for mk in A["micros"]:
        x, y = A["micros"][mk], B["micros"][mk]
        pose = mk.startswith("pose")
        r = {"v_identical": x["v"] == y["v"], "loss_identical": x["loss"] == y["loss"]}
        gs = [n for n in x["grad_sha"] if n != S]
        diff = [n for n in gs if x["grad_sha"][n] != y["grad_sha"].get(n)]
        vdiff = [n for n in gs if x["grad_vsha"][n] != y["grad_vsha"].get(n)]
        r.update(grads_compared=len(gs), grads_set=sum(x["grad_sha"][n] is not None for n in gs), grads_byte_differ=diff[:6], n_grads_byte_differ=len(diff),
                 grads_value_differ=vdiff[:6], n_grads_value_differ=len(vdiff),
                 dec_after_identical=x["dec_after_sha"] == y["dec_after_sha"], seconds=[x["seconds"], y["seconds"]])
        se = y.get("s_emb")
        r["s_emb"] = se
        if expect in ("same", "s2p_zero"):
            m_ok = r["v_identical"] and r["loss_identical"] and not diff and r["dec_after_identical"]
            if expect == "s2p_zero":
                if pose:
                    m_ok &= se is not None and se.get("grad_norm", 0) > 0 and se.get("nonzero_rows_not_present") == 0 and se.get("present_rows_zero") == 0
                else:
                    m_ok &= se == {"grad": None}
        else:                                                      # s2p_nz: only pose micros change
            if pose:
                vb, vo = x["v_float"], y["v_float"]
                m_ok = vb[15] != vo[15] and x["loss"] != y["loss"]            # pose cb0 CE sum (index 15 = len(TK) + 5) changed
                r["pose_cb0_sum"] = [vb[15], vo[15]]
                r["speech_cb0_sum_identical"] = vb[10] == vo[10]
                r["text_word_sums_identical"] = vb[:10] == vo[:10]
                m_ok &= r["speech_cb0_sum_identical"] and r["text_word_sums_identical"]
            else:                                                  # (dec_p already differs from the earlier pose micros' nanosteps:
                dsame = all(x["dec_after_sha"][n] == y["dec_after_sha"][n] for n in x["dec_after_sha"] if n.startswith("dec_s."))   # compare dec_s)
                r["dec_s_after_identical"] = dsame
                m_ok = r["v_identical"] and r["loss_identical"] and not diff and dsame and se == {"grad": None}
        r["ok"] = bool(m_ok)
        ok &= bool(m_ok)
        res["micros"][mk] = r
        print(f"  {mk:12s} {'ok ' if m_ok else 'BAD'} v== {r['v_identical']} loss== {r['loss_identical']} grads byte-differ "
              f"{len(diff)}/{len(gs)} (set {r['grads_set']}) value-differ {len(vdiff)} dec== {r['dec_after_identical']} "
              + (f"s_emb {se}" if se is not None else "") + (f" pose_cb0_sum {r.get('pose_cb0_sum')}" if 'pose_cb0_sum' in r else ""))
    res["PASS"] = bool(ok)
    json.dump(res, open(outp, "w"), indent=1)
    print(("PASS" if ok else "FAIL"), f"{expect}: {res['base']}  vs  {res['other']}")


if __name__ == "__main__":
    {"cfg": cfg, "var": var}[sys.argv[1]](sys.argv[2:])
