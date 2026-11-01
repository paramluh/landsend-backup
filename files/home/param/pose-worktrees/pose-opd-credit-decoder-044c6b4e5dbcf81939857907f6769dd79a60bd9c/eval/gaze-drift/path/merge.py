import json
P = "/data/home/amartyavarma/ppm/gaze-drift/path"
r = json.load(open(f"{P}/results.json"))
a = json.load(open(f"{P}/check_a.json"))["rows"]
r["goalA_render_path"] = {
    "verdict": "identical render path; only differences = length (125 vs 75 latent frames) and generated pose/audio after 2 s",
    "code": "GT (s20/vrefs/GT-<clip>, rendered by vrefs_worker.claim_one) and generated (ppm-pose-server video_worker) BOTH call vrefs_lib.Renderer.render -> "
            "demo-v3 Sampler.run(sample, seed) with prefill=0 (no video teacher forcing in either); same 9-sinks model, 40 stab + 10 res steps, "
            "audio guidance 2.0 on first 10 steps (both), seed 2026 (both), refs = load_refs(rng[1234, crc32(ref_seed_key)]) same key -> same picks; "
            "missing slots (S-20, E-18) built from the same FILL cache frames [75,150,225]. Window [9 refs | sink | 3 history | 4 present] with slot-relative RoPE: "
            "no absolute-position growth; no re-anchoring beyond sink+refs in either. vrefs.py (standalone) is NOT what rendered the GT cells.",
    "n_rows_checked": len(a),
    "frame0_pose_max_abs_diff": max(v["frame0_pose_maxabs"] for v in a.values()),
    "prefill_tokens_eq_GT_all": all(v["prefill_eq_pkg_gt_tokens"] and v["prefill_eq_run_gt_pose"] for v in a.values()),
    "prefill_mimi_eq_GT_all": all(v["audio_prefill_eq_gt_mimi_local"] for v in a.values()),
    "global_state_eq_all": all(v["gen_global_state"] == v["pkg_global_state"] for v in a.values()),
    "sink_bit_equal_all": all(v["sink_gen_eq_gt"] and v["sink_gen_eq_pkg_sink"] for v in a.values()),
    "prefill_pose_latents_max_rel_diff": round(max(v["prefill_latents_max_rel_diff"] for v in a.values()), 4),
    "post_prefill_pose_latents_mean_rel_diff": round(sum(v["post_latents_mean_rel_diff"] for v in a.values()) / len(a), 3),
    "latent_frames_gen_vs_gt": [125, 75],
    "status_json_diffs": "only latent/decoded frames + duration (and refs metadata wording for the 2 built-slot clips)",
    "video_model_split_of_GT_clips": {"train": ["10_27286bdd", "06_74807fe0", "03_b437950e", "05_e25c6859", "09_d9ef3c94", "12_72cb45d3", "13_956c09af"],
                                      "unseen_eval": ["19_47473a46", "15_d1e1322c", "18_1ffa9083"]},
    "pixdiff_GT_vs_gen_render_8bit_by_0.4s": "3.4-6.5 during 0-2 s (identical inputs; residual = 4-frame lookahead into generated latents + compile), ~15-17 after 2 s",
    "per_row": a,
}
json.dump(r, open(f"{P}/results.json", "w"), indent=1)
print("ok", r["goalA_render_path"]["frame0_pose_max_abs_diff"], r["goalA_render_path"]["prefill_tokens_eq_GT_all"], r["goalA_render_path"]["sink_bit_equal_all"])
