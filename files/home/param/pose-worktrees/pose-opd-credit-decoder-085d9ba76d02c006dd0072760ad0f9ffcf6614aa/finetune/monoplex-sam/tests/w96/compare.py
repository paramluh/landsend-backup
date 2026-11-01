"""w96 compare (CPU): step_equiv.py records -> layout difference (W' with dummy ranks vs W, all ranks real) next to the run-to-run noise
floor (same layout twice). usage: python compare.py <dir with W3_r0.pt ...> [ref tag] [dummy tag]  -> prints + out/compare.json
Metrics per optimizer group on the world-invariant subsamples (global flat index = off mod 997): rel L2 ||a - b|| / ||a||, max |a - b|,
fraction bit-identical; for parameters the UPDATE u = p_after - p_init is compared (||u_a - u_b|| / ||u_a||); scalars: relative differences."""
import json
import os
import sys

import numpy as np
import torch

D = sys.argv[1] if len(sys.argv) > 1 else "/scratch/amartyavarma/w96test"
REF, DUM = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("W3", "W4")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def group_of(n):                                                  # = model_ppm.group_of (+ decoders)
    if n.startswith("inp.emb.") or n.startswith("layers.") or n in ("head.norm.weight", "head.lm_head.weight"):
        return "trunk"
    if n in ("head.spec_out", "head.nw_out", "head.eob_out"):
        return "spec"
    if n.startswith("head.proj."):
        return "proj"
    if n.startswith("inp.aud.") or n.startswith("inp.pose."):
        return "audio"
    for g in ("decp_bb", "tower_s", "tower_p", "dec_s", "dec_p"):
        if n.startswith(g + "."):
            return g
    raise KeyError(n)


def load(tag, k):
    return torch.load(f"{D}/{tag}_r{k}.pt", map_location="cpu", weights_only=False)


R = {(t, k): load(t, k) for t in (REF, DUM) for k in (0, 1) if os.path.isfile(f"{D}/{t}_r{k}.pt")}
init = {t: R[(t, 0)]["fp_init"] for t in (REF, DUM)}
dec0 = {t: R[(t, 0)]["fp_dec0"] for t in (REF, DUM)}


def vec_cmp(fa, fb, keys, base_a=None, base_b=None):
    """per group: rel L2 of (a - b) vs a, max abs, frac bit-equal; with bases: on the updates a - base_a, b - base_b."""
    out = {}
    for g in sorted({group_of(n) for n in keys}):
        ns = sorted(n for n in keys if group_of(n) == g)
        a = np.concatenate([fa[n]["val"].astype(np.float64) for n in ns]) if ns else np.zeros(0)
        b = np.concatenate([fb[n]["val"].astype(np.float64) for n in ns]) if ns else np.zeros(0)
        for n in ns:
            assert np.array_equal(fa[n]["idx"], fb[n]["idx"]), ("subsample index mismatch", n)
        eq = float(np.mean(np.concatenate([fa[n]["val"].view(np.uint32) == fb[n]["val"].view(np.uint32) for n in ns]))) if len(a) else 1.0
        if base_a is not None:
            a = a - np.concatenate([base_a[n]["val"].astype(np.float64) for n in ns])
            b = b - np.concatenate([base_b[n]["val"].astype(np.float64) for n in ns])
        na = float(np.linalg.norm(a))
        out[g] = {"n": int(len(a)), "rel_l2": float(np.linalg.norm(a - b)) / na if na > 0 else (0.0 if not np.any(a - b) else float("inf")),
                  "max_abs": float(np.max(np.abs(a - b))) if len(a) else 0.0, "frac_bit_equal": eq, "norm_a": na,
                  "sumsq_rel": abs(sum(fa[n]["sumsq"] for n in ns) - sum(fb[n]["sumsq"] for n in ns)) / max(sum(fa[n]["sumsq"] for n in ns), 1e-300)}
    return out


def rel(a, b):
    return abs(a - b) / max(abs(a), 1e-30)


def pair(A, B, ta, tb):
    res = {"eval_raw_max_rel": max(rel(x, y) for x, y in zip(sum(A["eval_raw"], []), sum(B["eval_raw"], [])) if abs(x) > 0),
           "eval": {k: rel(A["eval"][k], B["eval"][k]) for k in A["eval"] if k.endswith(("word_ce_norm", "cb0_ce", "light_ce", "pose_cb0_ce", "pose_light_ce"))},
           "losscheck_rank0_packed_rel": [rel(x, y) for x, y in zip(A["losscheck"]["rank0_packed"], B["losscheck"]["rank0_packed"])],
           "steps": []}
    for sa, sb in zip(A["steps"], B["steps"]):
        s = {"step": sa["step"], "tot_equal": sa["tot"] == sb["tot"], "dec_M_equal": [sa["dec_s_M"] == sb["dec_s_M"], sa["dec_p_M"] == sb["dec_p_M"]],
             "lr_equal": sa["lr"] == sb["lr"] and sa["dec_s_lrs"] == sb["dec_s_lrs"] and sa["dec_p_lrs"] == sb["dec_p_lrs"],
             "scalars_rel": {k: rel(sa["d"][k], sb["d"][k]) for k in ("train/loss", "train_speech/word_ce_norm", "train_speech/pad_ce", "train_speech/cb0_ce",
                                                                     "train_speech/light_ce", "train_speech/pose_cb0_ce", "train_speech/pose_light_ce")},
             "gn_rel": {k: rel(sa["gn"][k], sb["gn"][k]) for k in sa["gn"]},
             "grad": vec_cmp(sa["fp_grad"], sb["fp_grad"], sa["fp_grad"].keys()),
             "update": vec_cmp(sa["fp_param"], sb["fp_param"], sa["fp_param"].keys(), init[ta], init[tb]),
             "param_bit_equal": vec_cmp(sa["fp_param"], sb["fp_param"], sa["fp_param"].keys()),
             "dec_update": vec_cmp(sa["fp_dec"], sb["fp_dec"], [n for n in sa["fp_dec"] if sa["fp_dec"][n]["val"].dtype == np.float32 and "rotary" not in n],
                                   dec0[ta], dec0[tb])}
        res["steps"].append(s)
    return res


out = {"init_identical_across_worlds": all(np.array_equal(init[REF][n]["val"], init[DUM][n]["val"]) and init[REF][n]["sum"] == init[DUM][n]["sum"]
                                           for n in init[REF]) and all(np.array_equal(dec0[REF][n]["val"], dec0[DUM][n]["val"]) for n in dec0[REF]),
       "roles": {t: R[(t, 0)]["roles"] for t in (REF, DUM)}, "worlds": {t: R[(t, 0)]["world"] for t in (REF, DUM)}}
pairs = [("noise_" + REF, (REF, 0), (REF, 1)), ("noise_" + DUM, (DUM, 0), (DUM, 1)), ("layout_r0", (REF, 0), (DUM, 0)), ("layout_r1", (REF, 1), (DUM, 1)),
         ("layout_x", (REF, 0), (DUM, 1))]
for name, ka, kb in pairs:
    if ka in R and kb in R:
        out[name] = pair(R[ka], R[kb], ka[0], kb[0])
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(f"{OUT}/compare_{REF}_{DUM}.json", "w"), indent=1)
print(f"init (resharded 128 -> each world) identical across worlds: {out['init_identical_across_worlds']}; roles {out['roles']}")
for name, _, _ in pairs:
    if name not in out:
        continue
    p = out[name]
    print(f"\n== {name}: eval raw sums max rel {p['eval_raw_max_rel']:.2e}; loss-check rank 0 packed rel {max(p['losscheck_rank0_packed_rel']):.2e}")
    for s in p["steps"]:
        print(f"  step {s['step']}: tot equal {s['tot_equal']} dec M equal {s['dec_M_equal']} lrs equal {s['lr_equal']}")
        print("    scalars rel: " + " ".join(f"{k.split('/')[-1]}={v:.2e}" for k, v in s["scalars_rel"].items()))
        print("    gn rel: " + " ".join(f"{k}={v:.2e}" for k, v in s["gn_rel"].items()))
        for kind in ("grad", "update", "dec_update"):
            print(f"    {kind} rel_l2: " + " ".join(f"{g}={v['rel_l2']:.2e}" for g, v in s[kind].items()))
        print("    param frac bit-equal: " + " ".join(f"{g}={v['frac_bit_equal']:.4f}" for g, v in s["param_bit_equal"].items()))
