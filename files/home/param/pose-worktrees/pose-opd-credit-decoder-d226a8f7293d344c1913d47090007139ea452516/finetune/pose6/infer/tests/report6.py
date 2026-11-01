#!/usr/bin/env python3
"""render tests/out/engine6_results_<job>.json as the TESTS.md GPU tables (markdown to stdout). usage: report6.py <json>"""
import json
import sys

R = json.load(open(sys.argv[1]))


def cell(d):
    """diff dict -> '0 (all identical)' or the non-zero entries."""
    nz = {k: v for k, v in d.items() if v}
    return "0 (all identical)" if not nz else "; ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in nz.items())


out = []
if "i" in R:
    out += ["### (i) flags off and (ii) zero new params vs the DEPLOYED engine (one process, orig-vs-orig floor first)", "",
            "| mode / scenario | frames | floor orig vs orig | (i) orig vs off | (ii) orig vs s2p cb0 (SE = 0) | (ii) orig vs s2p cb0 + G0COND (SE = 0, G0 = 0, f = F_A) | g0 rows applied | pass |",
            "|---|---|---|---|---|---|---|---|"]
    for k, r in R["i"].items():
        out.append(f"| {k} | {r['frames']} | {cell(r['floor_orig_vs_orig'])} | {cell(r['i_orig_vs_off'])} | "
                   f"{cell(r['ii_orig_vs_s2p_zero']) if 'ii_orig_vs_s2p_zero' in r else '(graphs only)'} | {cell(r['ii_orig_vs_new_zero_fA'])} | "
                   f"{[(a, b) for a, b, _ in r['g0_rows_new']]} | {r['pass']} |")
    out += ["", f"(i)/(ii) all pass: **{R.get('i_pass')}**. Compared per frame: B lane, speech codes, pose codes (readback) + hv_p, pose cb0 logits "
            "(both rows), hv_s, speech cb0 logits, hn (both rows); 0 = bit-identical.", ""]
if "scale" in R:
    s = R["scale"]
    out += ["### non-zero parameter scales (used in (w), (iii), (iv))", "",
            f"- inp.g0 ~ N(0, sigma^2), sigma = {s['sigma']:.4g}: chosen so ||g0(F_A)|| = the median per-frame ||inp.pose(pose codes)|| of Person 01's "
            f"opening = {s['pose_frame_input_norm_median']:.3g} (p10-p90 {s['pose_frame_input_norm_p10_p90'][0]:.3g}-{s['pose_frame_input_norm_p10_p90'][1]:.3g}); "
            f"realised ||g0(F_A)|| = {s['g0_F_A_norm']:.3g}, ||g0(F_B)|| = {s['g0_F_B_norm']:.3g} (||F_A|| {s['norm_F_A']:.3g}, ||F_B|| {s['norm_F_B']:.3g}); "
            f"for scale: inp.pose row norm median {s['inp_pose_row_norm_median']:.3g}, emb_B('\\n') row norm {s['emb_B_newline_row_norm']:.3g}.",
            f"- tower_p.s_emb ~ N(0, (0.02 / sqrt 32)^2) (= pose-s2p, the tower code_emb init scale): row norm median {s['s_emb_row_norm_median']:.3g}.",
            f"- F_A = {[round(x, 4) for x in R['f']['F_A']]} (Person 01 opening frame 0 root / width, 9:16), "
            f"F_B = {[round(x, 4) for x in R['f']['F_B']]}.", ""]
if "w" in R:
    w = R["w"]
    w1 = w["w1_prefill_locality_and_uncond_same_f"]
    out += ["### (w) G0COND wiring (non-zero G0, SE = 0)", "",
            f"- w1 prefill locality (P = {w1['P']}, SLOT = {w1['P'] - 1}), new vs off engine, trunk KV:"]
    for k, v in w1["locality"].items():
        out.append(f"  - {k}: K differs at {v['n_K_diff']} positions, first {v['K_differing_positions']}; V at {v['n_V_diff']}, first {v['V_differing_positions']}")
    out += [f"  - uncond row's layer-0 K/V at the SLOT == the cond row's (same f): {w1['uncond_row_SLOT_kv_layer0_equals_cond_row']}; "
            f"g0 rows applied {[(a, b) for a, b, _ in w1['g0_rows']]}; hn_prev change cond {w1['cond_hn_prev_maxdiff']:.3g}, uncond "
            f"{w1['uncond_hn_prev_maxdiff']:.3g}; pass {w1['pass']}"]
    for k in [k for k in w if k.startswith("w2")]:
        out.append(f"- {k}: {cell(w[k]['diff'])}; pass {w[k]['pass']}")
    w3 = w["w3_f_changes_output"]
    out += [f"- w3 f matters (graphs, free, 60 frames): F_A vs F_B {cell(w3['F_A_vs_F_B'])}; F_A vs off {cell(w3['F_A_vs_off'])}; F_A repeat floor "
            f"{cell(w3['F_A_repeat_floor'])}; frame-0 hn change vs off {w3['frame0_hn_maxdiff_F_A_vs_off']:.3g}; pass {w3['pass']}"]
    w4 = w["w4_fail_loud"]
    out += ["- w4 fail loud on the real engines: " + "; ".join(f"{k}: {v if v else 'no exception'}" for k, v in w4.items() if k != "pass")
            + f"; pass {w4['pass']}", "", f"(w) all pass: **{R.get('w_pass')}**", ""]
if "iii" in R:
    tm = R.get("iii_train_model", {})
    out += ["### (iii) engine vs the TRAINING forward, teacher-forced from conversation frame 0 (cond row + pose-CFG uncond row)", "",
            f"Training model: `{tm.get('file')}` (B1 port: {tm.get('b1_port')}; G0COND in the training class: {tm.get('g0_in_training_class')})"
            + (f"; {tm['note']}" if tm.get("note") else "") + ".", "",
            "| params / micro / mode | P | frames | text top1 / TV mean / max abs | speech cb0 top1 / max abs | pose cb0 cond top1 / TV / max abs | pose cb0 UNCOND top1 / TV / max abs | hn cond / uncond max abs (ref absmax) | hv_p / zb max abs | pose dec e2e top1 / max abs (same-input floor) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in R["iii"].items():
        t, s0, pc, pu = r["text"], r["speech_cb0"], r["pose_cb0_cond"], r["pose_cb0_uncond"]
        out.append(f"| {k} | {r['P']} | {r['frames']} | {t['top1']} / {t['tv_mean']:.2e} / {t['max_abs']:.3g} | {s0['top1']} / {s0['max_abs']:.3g} | "
                   f"{pc['top1']} / {pc['tv_mean']:.2e} / {pc['max_abs']:.3g} | {pu['top1']} / {pu['tv_mean']:.2e} / {pu['max_abs']:.3g} | "
                   f"{r['hn_cond_max_abs']:.3g} / {r['hn_uncond_max_abs']:.3g} ({r['hn_ref_absmax']}) | {r['hv_p_cond_max_abs']:.3g} / {r['zb_cond_max_abs']:.3g} | "
                   f"{r['pose_dec_e2e']['top1']} / {r['pose_dec_e2e']['max_abs']:.3g} ({r['pose_dec_same_input']['top1']} / {r['pose_dec_same_input']['max_abs']:.3g}) |")
    eff = [(k, r["effect_trainer_nonzero_vs_zero"]) for k, r in R["iii"].items() if "effect_trainer_nonzero_vs_zero" in r]
    if eff:
        out += ["", "Effect size of the mods in the TRAINER itself (non-zero vs zero new params, same micro): the parity errors above must be far below these.", "",
                "| micro / mode | hn max abs | text top1 / TV mean / max abs | pose cb0 top1 / TV mean / max abs | speech cb0 top1 / max abs |", "|---|---|---|---|---|"]
        for k, e in eff:
            out.append(f"| {k} | {e['hn_max_abs']:.3g} | {e['text']['top1']} / {e['text']['tv_mean']:.2e} / {e['text']['max_abs']:.3g} | "
                       f"{e['pose_cb0']['top1']} / {e['pose_cb0']['tv_mean']:.2e} / {e['pose_cb0']['max_abs']:.3g} | {e['speech_cb0']['top1']} / {e['speech_cb0']['max_abs']:.3g} |")
    out.append("")
if "iv" in R:
    out += ["### (iv) per-frame latency (same 2 GPUs, same process; sync = both GPUs synchronised every frame, median / p90; async = no per-frame sync)", "",
            "| mode / scenario | today (deployed engine) | s2p cb0 | s2p cb0 + G0COND | start() prefill ms (today / s2p / +G0) |", "|---|---|---|---|---|"]
    keys = sorted({k.rsplit("/", 1)[0] for k in R["iv"]}, key=lambda x: (x.split("/")[0] != "graphs", x))
    for k in keys:
        cs = [R["iv"].get(f"{k}/{lab}") for lab in ("today", "s2p_cb0", "s2p_cb0+g0")]
        fmt = lambda c: f"sync {c['sync_ms_median']} / p90 {c['sync_ms_p90']}; async {c['async_ms_per_frame']}" if c else "-"  # noqa: E731
        out.append(f"| {k} ({cs[0]['frames'] if cs[0] else '?'} frames) | {fmt(cs[0])} | {fmt(cs[1])} | {fmt(cs[2])} | "
                   f"{' / '.join(str(c['start_prefill_ms']) if c else '-' for c in cs)} |")
    out.append("")
print("\n".join(out))
