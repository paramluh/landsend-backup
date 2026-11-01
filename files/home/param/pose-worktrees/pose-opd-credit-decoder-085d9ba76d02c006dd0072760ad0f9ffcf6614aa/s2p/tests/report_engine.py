#!/usr/bin/env python3
"""engine_results*.json -> markdown sections (a) / (b) / (d) for TESTS.md (printed to stdout)."""
import json
import sys

OUT = "/data/home/amartyavarma/ppm/pose-s2p/tests/out"
r1 = json.load(open(f"{OUT}/engine_results_run1_j30894.json"))
r2 = json.load(open(f"{OUT}/engine_results.json"))
A = dict(r1.get("a", {}))
A.update(r2.get("a", {}))


def zero(d):
    return all(v == 0 for v in d.values())


print("## (a) exactness at init (deployed layout): " + ("PASS" if all(v["pass"] for v in A.values()) else "FAIL"))
print(f"\nJobs 30894 (eager, and the graphs free / free_cfg runs) and 30924 (graphs, all three scenarios) on slinky-10. GPUs: {r2.get('gpus', ['?'])[0]} x 2.")
print("Script `tests/test_engine.py`; raw results are in `tests/out/engine_results_run1_j30894.json` and `tests/out/engine_results.json`.\n")
print("**Setup.** One process. The model is export-357 on 2 GPUs (37 + 27 backbone layers; pose branch on cuda:0), loaded once and shared by four engines:")
print("- `orig`: the ORIGINAL deployed `aloe-pose-monoplex-infer/ppm/fast_ppm.py`, loaded under another module name. Every other dependency file is asserted byte-identical.")
print("- `off`: the branch engine with the flag off. The model has no s_emb, as with every deployed checkpoint.")
print("- `full`: the branch engine, s2p full, with zero s_emb.")
print("- `cb0`: the branch engine, s2p cb0, with zero s_emb.\n")
print("Each engine is built as genweb worker.py builds it: `FastPPM(n_inst=1, cfg_rows=True, max_s=133.68, mimi_stream=False, silence=Mimi(zeros))`, with worker.py's DEF params and seed 1234.")
print("The voice clip is the last 10 s of Person 01's opening.\n")
print("Each scenario runs `orig` twice; the second run is the floor. Every row compares against the first `orig` run.")
print("A cell shows the number of differing codes, or the max abs latent difference; 0 means bit-identical.\n")
print("| mode / scenario | frames | floor orig vs orig | orig vs off | orig vs full | orig vs cb0 | pass |")
print("|---|---|---|---|---|---|---|")
for k, v in A.items():
    cells = []
    for c in ("floor_orig_vs_orig", "orig_vs_off", "orig_vs_full", "orig_vs_cb0"):
        d = v[c]
        cells.append("0 (all identical)" if zero(d) else json.dumps(d))
    print(f"| {k} | {v['frames']} | " + " | ".join(cells) + f" | {v['pass']} |")
print("\nCompared per frame: B lane, speech codes, pose codes (readback), plus hv_p, pose cb0 logits (both rows), hv_s, speech cb0 logits and hn (both rows).")
print("- `free` and `free_cfg` are free generation. In `free_cfg`, `pose_cfg_on=1`, so the uncond rows feed the pose draw.")
print("- `poseonly` is worker.py's pose-only mode: real speech codes plus the timed-word lane are forced every frame, and pose is generated.\n")
B = r2.get("b", {})
if B:
    print("## (b) wiring, non-zero s_emb: " + ("PASS" if all(v["pass"] for v in B.values()) else "FAIL"))
    print("\nThe same process as (a)'s second job. s_emb is set to N(0, 0.02 / sqrt 32), the init scale of the tower code_emb, for both full and cb0.\n")
    print("**b1: pose forced to the flag-off run's pose; everything except pose must equal flag off.**\n")
    print("| mode / variant | lane, speech codes, hv_s, speech logits, hn vs flag off | pose cb0 logits changed (frames) | max change | pass |")
    print("|---|---|---|---|---|")
    for k, v in B.items():
        if "/b1_" in k:
            print(f"| {k.split('/')[0]} / {k.split('/')[-1]} | {'bit-identical' if v['speech_text_trunk_identical'] else 'DIFFERENT'} | "
                  f"{v['pose_logit_frames_changed']} of {v['of']} | {v['pose_logit_maxdiff']:.3g} | {v['pass']} |")
    print("\n**b2: B's speech codes at frame k perturbed (pose-only run).**\n")
    print("| mode / engine / perturbed | pose logits at k changed (expected) | max change at k | identical before k | speech identical through k | pose at k+1 changed | pass |")
    print("|---|---|---|---|---|---|---|")
    for k, v in B.items():
        if "/b2_" in k:
            mo, _, en, wh = k.split("/")
            print(f"| {mo} / {en} / {wh} | {v['pose_logits_changed_at_k']} ({v['expected_change_at_k']}) | {v['pose_logit_maxdiff_at_k']:.3g} | "
                  f"{v['pose_logits_identical_before_k']} | {v['speech_logits_identical_through_k']} | {v['pose_logits_changed_at_k+1 (via trunk)']} | {v['pass']} |")
    print("\n**b3: what s_emb reads.**\n")
    print("| variant | instance rows = B's codes at T | uncond rows = silence codes | prefill pose KV differs only from the clip's first coded position (cond, uncond) | pass |")
    print("|---|---|---|---|---|")
    for k, v in B.items():
        if "/b3_" in k:
            print(f"| {k.split('/')[-1]} | {v['cond_rows_read_B_codes_at_T']} | {v['uncond_rows_read_silence_codes']} | "
                  f"{v['prefill_kv_differs_only_from_clip_on (cond, uncond)']} (clip from position {v['first_clip_pos']}, P {v['P']}) | {v['pass']} |")
    print("\nThe three tests:")
    print("- **b1.** A free run where pose is forced to the flag-off run's pose codes. Every speech, text and trunk output must stay bit-identical to flag off, while the pose cb0 logits change on every frame.")
    print("- **b2.** In the pose-only run, B's speech codes at frame k are perturbed (all codebooks, cb0 only, or cb1..31 only). Expected:")
    print("  - flag off: pose at k is unchanged;")
    print("  - full: pose at k changes for every perturbation;")
    print("  - cb0: pose at k changes only when cb0 changes;")
    print("  - every engine: speech at k and before is unchanged, and pose at k+1 changes via the trunk.")
    print("- **b3.** s_emb's inputs: instance rows must read B's speech codes at T, and uncond pose-CFG rows the Mimi codes of digital silence. The prefill pose-tower KV must differ from flag off only from the voice clip's first coded position on, in both rows.\n")
D = r2.get("d", {})
if D:
    print("## (d) per-frame latency: CUDA graphs, same 2 GPUs, same process\n")
    print("Each cell is ms per frame over 300 frames, or the opening's length for poseonly. Two numbers per cell:")
    print("- **sync**: the frame is synchronised on both GPUs after every step (the live-demo pattern); median / p90.")
    print("- **async**: no per-frame sync (genweb worker's loop); total time / frames.\n")
    print("| scenario | today (orig, pose concurrent with speech) | s2p full (serial speech -> pose) | s2p cb0 (pose after the speech cb0 draw) |")
    print("|---|---|---|---|")
    for sc in ("free", "free_cfg", "poseonly"):
        cells = []
        for nm in ("orig", "full", "cb0"):
            x = D.get(f"{sc}/{nm}")
            cells.append("-" if x is None else f"sync {x['sync_ms_median']} / p90 {x['sync_ms_p90']}; async {x['async_ms_per_frame']}")
        print(f"| {sc} | " + " | ".join(cells) + " |")
    print()
