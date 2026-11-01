# pose-s2p tests

**Summary (10-27):**
- **(a) exactness at init: PASS.** All bit-identical, including the old-vs-old floor (= 0), eager and graphs.
- **(b) wiring: PASS.**
- **(c) training path: PASS.**
- **(d) latency:** today 61.0 ms/frame, s2p full 96.3 ms, s2p cb0 66.3 ms.
- **(e) dry launch config:** ready, NOT submitted.

## (a) exactness at init (deployed layout): PASS

Jobs 30894 (eager, and the graphs free / free_cfg runs) and 30924 (graphs, all three scenarios) on slinky-10. GPUs: NVIDIA H100 80GB HBM3 x 2.
Script `tests/test_engine.py`; raw results are in `tests/out/engine_results_run1_j30894.json` and `tests/out/engine_results.json`.

**Setup.** One process. The model is export-357 on 2 GPUs (37 + 27 backbone layers; pose branch on cuda:0), loaded once and shared by four engines:
- `orig`: the ORIGINAL deployed `aloe-pose-monoplex-infer/ppm/fast_ppm.py`, loaded under another module name. Every other dependency file is asserted byte-identical.
- `off`: the branch engine with the flag off. The model has no s_emb, as with every deployed checkpoint.
- `full`: the branch engine, s2p full, with zero s_emb.
- `cb0`: the branch engine, s2p cb0, with zero s_emb.

Each engine is built as genweb worker.py builds it: `FastPPM(n_inst=1, cfg_rows=True, max_s=133.68, mimi_stream=False, silence=Mimi(zeros))`, with worker.py's DEF params and seed 1234.
The voice clip is the last 10 s of Person 01's opening.

Each scenario runs `orig` twice; the second run is the floor. Every row compares against the first `orig` run.
A cell shows the number of differing codes, or the max abs latent difference; 0 means bit-identical.

| mode / scenario | frames | floor orig vs orig | orig vs off | orig vs full | orig vs cb0 | pass |
|---|---|---|---|---|---|---|
| eager/free | 100 | 0 (all identical) | 0 (all identical) | 0 (all identical) | 0 (all identical) | True |
| eager/free_cfg | 100 | 0 (all identical) | 0 (all identical) | 0 (all identical) | 0 (all identical) | True |
| eager/poseonly | 100 | 0 (all identical) | 0 (all identical) | 0 (all identical) | 0 (all identical) | True |
| graphs/free | 150 | 0 (all identical) | 0 (all identical) | 0 (all identical) | 0 (all identical) | True |
| graphs/free_cfg | 150 | 0 (all identical) | 0 (all identical) | 0 (all identical) | 0 (all identical) | True |
| graphs/poseonly | 149 | 0 (all identical) | 0 (all identical) | 0 (all identical) | 0 (all identical) | True |

Compared per frame: B lane, speech codes, pose codes (readback), plus hv_p, pose cb0 logits (both rows), hv_s, speech cb0 logits and hn (both rows).
- `free` and `free_cfg` are free generation. In `free_cfg`, `pose_cfg_on=1`, so the uncond rows feed the pose draw.
- `poseonly` is worker.py's pose-only mode: real speech codes plus the timed-word lane are forced every frame, and pose is generated.

## (b) wiring, non-zero s_emb: PASS

The same process as (a)'s second job. s_emb is set to N(0, 0.02 / sqrt 32), the init scale of the tower code_emb, for both full and cb0.

**b1: pose forced to the flag-off run's pose; everything except pose must equal flag off.**

| mode / variant | lane, speech codes, hv_s, speech logits, hn vs flag off | pose cb0 logits changed (frames) | max change | pass |
|---|---|---|---|---|
| eager / full | bit-identical | 60 of 60 | 0.375 | True |
| eager / cb0 | bit-identical | 60 of 60 | 0.125 | True |
| graphs / full | bit-identical | 120 of 120 | 0.471 | True |
| graphs / cb0 | bit-identical | 120 of 120 | 0.188 | True |

**b2: B's speech codes at frame k perturbed (pose-only run).**

| mode / engine / perturbed | pose logits at k changed (expected) | max change at k | identical before k | speech identical through k | pose at k+1 changed | pass |
|---|---|---|---|---|---|---|
| eager / off / all | False (False) | 0 | True | True | True | True |
| eager / off / cb0 | False (False) | 0 | True | True | True | True |
| eager / off / light | False (False) | 0 | True | True | True | True |
| eager / full / all | True (True) | 0.0938 | True | True | True | True |
| eager / full / cb0 | True (True) | 0.0625 | True | True | True | True |
| eager / full / light | True (True) | 0.0938 | True | True | True | True |
| eager / cb0 / all | True (True) | 0.0625 | True | True | True | True |
| eager / cb0 / cb0 | True (True) | 0.0625 | True | True | True | True |
| eager / cb0 / light | False (False) | 0 | True | True | True | True |
| graphs / off / all | False (False) | 0 | True | True | True | True |
| graphs / off / cb0 | False (False) | 0 | True | True | True | True |
| graphs / off / light | False (False) | 0 | True | True | True | True |
| graphs / full / all | True (True) | 0.156 | True | True | True | True |
| graphs / full / cb0 | True (True) | 0.0625 | True | True | True | True |
| graphs / full / light | True (True) | 0.156 | True | True | True | True |
| graphs / cb0 / all | True (True) | 0.0625 | True | True | True | True |
| graphs / cb0 / cb0 | True (True) | 0.0625 | True | True | True | True |
| graphs / cb0 / light | False (False) | 0 | True | True | True | True |

**b3: what s_emb reads.**

| variant | instance rows = B's codes at T | uncond rows = silence codes | prefill pose KV differs only from the clip's first coded position (cond, uncond) | pass |
|---|---|---|---|---|
| full | True | True | [True, True] (clip from position 6, P 134) | True |
| cb0 | True | True | [True, True] (clip from position 6, P 134) | True |

The three tests:
- **b1.** A free run where pose is forced to the flag-off run's pose codes. Every speech, text and trunk output must stay bit-identical to flag off, while the pose cb0 logits change on every frame.
- **b2.** In the pose-only run, B's speech codes at frame k are perturbed (all codebooks, cb0 only, or cb1..31 only). Expected:
  - flag off: pose at k is unchanged;
  - full: pose at k changes for every perturbation;
  - cb0: pose at k changes only when cb0 changes;
  - every engine: speech at k and before is unchanged, and pose at k+1 changes via the trunk.
- **b3.** s_emb's inputs: instance rows must read B's speech codes at T, and uncond pose-CFG rows the Mimi codes of digital silence. The prefill pose-tower KV must differ from flag off only from the voice clip's first coded position on, in both rows.

## (d) per-frame latency: CUDA graphs, same 2 GPUs, same process

Each cell is ms per frame over 300 frames, or the opening's length for poseonly. Two numbers per cell:
- **sync**: the frame is synchronised on both GPUs after every step (the live-demo pattern); median / p90.
- **async**: no per-frame sync (genweb worker's loop); total time / frames.

| scenario | today (orig, pose concurrent with speech) | s2p full (serial speech -> pose) | s2p cb0 (pose after the speech cb0 draw) |
|---|---|---|---|
| free | sync 60.97 / p90 61.13; async 60.58 | sync 96.3 / p90 96.5; async 95.68 | sync 66.27 / p90 66.5; async 65.87 |
| free_cfg | sync 64.85 / p90 65.04; async 64.45 | sync 100.27 / p90 100.48; async 99.44 | sync 70.06 / p90 70.27; async 69.64 |
| poseonly | sync 61.43 / p90 61.56; async 60.86 | sync 96.73 / p90 96.89; async 95.69 | sync 66.5 / p90 66.65; async 65.91 |

**Reading against the 80 ms live budget (12.5 Hz):**
- **today:** 61.0 ms (65 ms with pose CFG on).
- **s2p full:** 96.3 ms (100.3 ms with CFG). That is **+35 ms, over budget**: pose now waits for the whole 31-codebook speech decoder.
- **s2p cb0:** 66.3 ms (70.1 ms with CFG). That is **+5.3 ms, within budget**: pose waits only for the speech tower and the cb0 draw, and the speech light decoder runs concurrently with pose, as today.

The pose-only runs show the same numbers. The genweb playground is offline, so it would run at the async rate: 1.6x real time for full, 1.2x for cb0.

## (c) training path: PASS (job 30876, CPU on slinky-18, 10-27 05:26-05:34 PT)

**Test:** `tests/test_train_path.py` (sbatch `tests/test_train_path.sbatch`). Results are in `tests/out/test_train_path.json` and the log is `tests/logs/test_train_path-30876.log`.

**Micro.** One real pose micro, built by the branch's own data code: `data_aloe_pose.Stores.conv` + `pack_micro` + the trainer's `prep`, with the mono8020 env.
- The item is eval item (store 0, row 698, swap 0), `ALOE/s2handover-pose/01a4adf4-.../seg00/c00`. It is the shortest eval item that has B's voice clip in the system block.
- P = 70 and the micro has **7,572 positions**:
  - 7,561 positions carry B speech codes: **61 in the voice clip** in the system block (where there is no pose) and 7,500 in the conversation;
  - 7,500 positions carry pose labels, the first at P + CD = 72;
  - every pose-labelled position T has speech codes at the same T.

**Model.** A tiny PPM, bf16, random init:
- backbone: 2 layers, H 128;
- towers: 2 layers, H 64;
- decoders: 1 layer;
- real vocab 151,936 and tower vocab 151,682.

The GPU-only kernels are replaced, in this test only, by torch reference code. That covers fused rms, add_rms, qk_prep and silu_mul, FA3 varlen (as per-segment causal SDPA), and the CE head's `out_dtype` GEMMs. Both sides of every comparison run the same replacement.

| check | result |
|---|---|
| c0: branch flag OFF vs the ORIGINAL `aloe-pose-rosie/code/model_ppm.py`, same weights | Every forward term (text, cb0, pcb0 and its sums, hv_s, hv_p, zb_p) is bit-identical (max abs 0). All 126 parameter grads are bit-identical, both for the pose-cb0 backward and for the run_micro total backward. |
| c1 `full`: flag ON with s_emb = 0 vs OFF | All terms bit-identical (max abs 0.0). All 126 other grads bit-identical. **s_emb grad norm 0.199** (pose cb0 loss only) and **15.6** (total). Rows with a non-zero grad: 46,738 of 65,536, exactly the 46,738 distinct speech codes present in B's depth. Rows of absent codes are exactly 0. |
| c1 `cb0`: flag ON with s_emb = 0 vs OFF | All terms bit-identical. All other grads bit-identical. **s_emb grad norm 0.0251** (pose only) and **2.78** (total). Rows with a non-zero grad: 1,417 of 2,048, exactly the distinct cb0 codes present. |
| c2: non-zero s_emb, N(0, 0.02 / sqrt 32) | Every speech and text term (text, cb0, hv_s and its sums) is bit-identical to flag off: speech never depends on s_emb. The pose terms change: pcb0 7.62042 -> 7.61968 (full) and -> 7.62032 (cb0); hv_p max change 0.43 (full) and 0.070 (cb0). Before the first speech-coded position (6, the voice clip), hv_p is unchanged. At that position it changes. |

**How the "total" backward is formed.** It is run_micro's loss: lw_word x text + cb0 + pose cb0, where lw_word = 10 is the mono8020 per-stream multiplier on the word lane. The decoders' upstream gradients into hv_s, hv_p and zb_p enter as fixed seeded tensors, through the same `torch.autograd.backward(roots, grads)` call that run_micro uses.

**Design note.** s_emb is a zero-initialised code table. The table is itself the term's output projection: it is a linear map of the one-hot codes. So step 0 equals today's model exactly, and every used row gets a gradient from step 0. A random table followed by a zero output projection would instead give the table exactly zero gradient at step 0.

**Training-side bug fixed during review.** In `train_aloe_pose.load_init`, the row loop looked `tower_p.s_emb.weight` up in the source shards, which would raise KeyError, before the parent patch's zero-fill could run. The loop now skips that key when the init lacks it, and the zero-fill then applies.

**Other notes.**
- s_emb sits in group tower_p, so it shares tower_p's LR (1.5e-5) and grad clip (1.0).
- Because the clip is computed per group, the new rows' gradient enters tower_p's clip norm. Once clipping is active, that can scale the other tower_p updates slightly differently from a flag-off run. This is inherent to adding a parameter to the group.

## (e) dry launch config, NOT submitted (10-27 05:40 PT)

**Files** (all under `pose-s2p/`):
- `s2p_mono8020.env`: `aloe-pose-mono-8020/aloe_mono8020.env` verbatim, apart from the lines below.
- `s2p_mono8020_run.sh`: `aloe_mono8020_run.sh` pointed at `pose-s2p/code` and the new env.
- `s2p_mono8020.sbatch`.

**Init.** PPM local-11488, the same init as mono8020. The result is therefore directly comparable to export-357; the only differences are s2p and world 8.

**Launch command (do not run until main / user decide):**
`sbatch /data/home/amartyavarma/ppm/pose-s2p/s2p_mono8020.sbatch`
- It requests 1 node `-w slinky-10`, 8 tasks x 4 CPUs, 8 GPUs, `--mem=530G`, `-t 1-06:00:00`. To use slinky-18 instead, add `-w slinky-18`.
- The run script only allows slinky-10 or slinky-18, with one node.
  - slinky-9 is never used, and the MoT job 30289's nodes are excluded.
  - slinky-4 is excluded because its GPUs 6-7 thermal-throttle.

### `diff aloe_mono8020.env s2p_mono8020.env`, key lines (comments omitted)
```
< QQA_RUN=ppm-aloe-pose-mono-8020                                   > QQA_RUN=ppm-pose-s2p-mono8020
< QQA_LOG_ROOT / QQA_PLAN_DIR / QQA_CONTROL_DIR = aloe-pose-mono-8020/{runs,plan,control}   > pose-s2p/{runs,plan,control}
< QQA_EXPORT=.../aloe-pose-mono-8020/export-357                     > QQA_EXPORT=.../pose-s2p/export-357
< ALOE_LOCAL_EXPORT_ROOT=/scratch/amartyavarma/aloe-pose-mono-8020/exports   > /scratch/amartyavarma/pose-s2p-mono8020/exports
< ALOE_SAVE_ROOT=/scratch/amartyavarma/aloe-pose-mono-8020/saves   > /scratch/amartyavarma/pose-s2p-mono8020/saves
< ALOE_SAVE_EVERY_MIN=60                                            > ALOE_SAVE_EVERY_MIN=100000      [world 8, = Rosie]
                                                                    > PPM_S2P=1                       [THE change]
                                                                    > ALOE_OPT_OFFLOAD=1              [world 8, = Rosie]
                                                                    > ALOE_OPT_OFFLOAD_CHUNK_GB=2     [world 8, = Rosie]
                                                                    > ALOE_SKIP_FINAL_FULL=1          [world 8, = Rosie]
```
Every other line is identical. That includes the LRs, clips, weights, steps 357, warmup 12, export steps 60/119/178/238/298, 64 rows per step, ALOE_ROWS_DIGEST and the prompt rule.

### Resolved-config diff (CPU dry mode: `tests/dry_cfg.py` + `tests/dry_cfg.sbatch`, job 30875)
The dry mode builds the trainer's module-level CFG with no dist and no GPU.

**The dry mode is faithful.** Run on the ORIGINAL mono8020 code and env at world 64, it reproduces export-357's `meta.json` cfg with **0 keys different**.

**Branch code + `s2p_mono8020.env` at world 8 vs export-357's cfg: 10 keys differ.**
- 7 are names and paths: control_dir, export, local_export_root, log_root, plan_dir, run_tag, save_root.
- `s2p`: absent -> `full`.
- `save_every_min`: 60 -> 100000.
- `micro_layout`: `row_per_rank` -> `lpt`. This follows automatically from ROWS_PER_STEP 64 > WORLD 8.

ALOE_OPT_OFFLOAD and ALOE_SKIP_FINAL_FULL do not appear because they are env-only knobs, not CFG keys. The run script asserts them.

### World-8 plan (dry `A.make_plan_v2(8, cfg)`)
- 357 steps and 22,825 rows.
- **Rows digest 3681cc26ea97cad7**, the same as mono8020's. The digest covers items, rows and eval packing, so the data and the order of the step groups are identical.
- Micro layout `lpt`:
  - R per step is 6 to 10 rounds, mean 9.17. Over the run that is 3,274 rounds per rank, 26,192 micros in all, of which 212 are zero-weight dummies.
  - The maximum micro position bound is 9,502 (BUDGET).
- 529 eval micros.
- Maths: the tokens_step normalisers and the joint decoder nanosteps over the rounds are the same objective and the same update rule as world 64 (mono8020 README). The run is **not bit-identical** to a world-64 run: the fp summation order differs, and so does the per-round grouping of micros.

### Time estimate (1 node, world 8)
**Measured inputs:**
- mono8020 at world 64: 11.6 s per step mean (11.4 s micro, 1 row per rank, ~6.1k positions).
- Rosie FT at world 8, same code, offload on: 15.3-19.6 s per step (micro 12.5-16.7 s, about 6.1k-position rows); peak 70.8 GiB per GPU.

**Training steps:** 9.17 rounds x 11.4-15.6 s + ~3 s for the offloaded AdamW step ≈ 108-146 s per step. Over 357 steps that is **~10.7-14.5 h**.

**Evals and exports:** 15 evals at about 3 min each at world 8 (66 eval micros per rank; mono8020 took about 25 s for 9), plus 5 node-local exports and the final export. Together ≈ 1 h.

**Total ≈ 12-16 h, central ~14 h.** Two nodes (world 16) would be about half as long.

### Risks / decisions for main
1. **No resumable saves at world 8.** This is the Rosie finding: a FULL save gathers the masters plus m/v to host on top of the 308 GB of pinned m/v, and that runs out of host RAM. So ALOE_SAVE_EVERY_MIN is 100000. STOP_AFTER_SAVE would hit the same OOM. A crash in a ~14 h run loses everything after the last weights-only export (steps 60/119/178/238/298 carry no AdamW state).
   - Option: world 16 (2 nodes), which halves host m/v per node. It may even fit without offload; that is unverified.
2. **/scratch space.** The run needs 6 node-local exports of about 88 GB each, roughly 530 GB. The run script's NEED is 540 GB and it aborts before training if that is not free. Free /scratch today: slinky-10 495 GB, slinky-4 73 GB, slinky-18 31 GB, so **no spare node passes**. Options:
   - (a) set ALOE_LOCAL_EXPORT_ROOT to /data, as the Rosie run did (/data has 1,238 GB free and is 96% full);
   - (b) free about 50 GB on slinky-10's /scratch;
   - (c) drop some intermediate exports. That changes no maths, but it is a recipe change.
3. **Nodes.** All 24 spare GPUs are serving demos. slinky-10 is the recommended node: its rail-7 problem passed re-test on 10-24, and for an 8-GPU job everything stays inside the node. slinky-18 is the misc node; the Rosie run used it.
4. **s_emb sits in group tower_p (same LR and clip).** Its gradient enters tower_p's clip norm, so once clipping is active the other tower_p updates can differ slightly from the flag-off recipe.
