# pose5 / code-811-s2p: the s2p port ("speech conditions pose") and its proofs

**S2P: PASS / DONE (10-27 07:40 PT) for BOTH PPM_S2P=1 (full) and PPM_S2P=cb0; flag off byte-identical to code-811.** Marker:
`code-811-s2p/tests/s2p/S2P_PASS` (JSON: variants passed + md5s of the certified files; written by `make_pass.py --write`). Workstream A
makes no further edits to code-811-s2p; D may merge G0COND now (then rerun this suite on the merged code). Nothing here launches training.

## What changed (copy only; ppm/code-811 untouched, its md5s in ../../ORIG_MD5.txt)
- `model_ppm.py`: `s2p_mode`, `S2PEmb` (= ppm/pose-s2p/code/model_ppm.py, same semantics), `Tower.forward(.., sdep=None)`,
  `PPM(.., s2p=None)` adds `tower_p.s_emb` only when on, `_towers` passes `sdep=b["depth"][1]` to the pose tower only when on,
  `init_rule`: `tower_p.s_emb.weight` -> zeros, `build(.., s2p=None)`.
- `train_ppm.py`: `PPM_S2P` (0/unset | 1/full | cb0) -> cfg keys `s2p`, `s2p_from_step` only when on; `build(.., s2p=)` only when on
  (+ startup assert / log line); `load()`: a checkpoint without s2p + flag on -> `allow_missing=("tower_p.s_emb.weight",)` (existing
  safety_ppm mechanism: kept at its zero init, no AdamW state), a checkpoint with s2p refuses flag off / the other mode;
  `s2p_from_step` set / carried; export `train_cfg.json` side info records `s2p` (flag on only); the ckpt round-trip build passes s2p.
- launch scripts in the copy now point at the copy (`ppm811.sbatch` C=, `ppm811f_watchdog.sh` / `ppm811_watchdog.sh` CODE=, their
  `.sbatch` exec paths), so nothing in the copy can silently relaunch the original code.
- No change to `s811.py`, `data_ppm.py`, `safety_ppm.py`, `fresh/`, `lib/`, `vendor/`: the s2p input (`depth[1]`) is already in every pose micro.

## Channel / row reasoning (8,1,1 rows)
- Pose rows (`data_ppm.pose_micro_ppm` -> `lib/pose_s3q.pose_micro`, used by fresh rows too): channel A is ABSENT (ids PAD, depth -1,
  present 0; prov `chan_A_role: absent`, `chan_B_role: assistant`); channel B carries the filmed person's word lane, Mimi codes
  (`depth[1]`) AND pose codes (`pdepth`). So "B's speech codes" on a pose row = the person's own audio, no ambiguity.
- Timing: `pose_s3q.segment_arrays`: `depth_B[q]` = Mimi of row frame `s0 + q - CD`, `pdepth[q]` = pose of frame `s0 + q - pose_delay`,
  `pose_delay = cfg pose_delay = CD = 2` (PPM_CD=2). The pose tower's label at T is `pdepth[T, 0]` (pose cb0 of frame T - CD) and it
  reads its own codes shifted (`shift1(pdepth)`); s_emb reads `depth[1][T]` UNSHIFTED = speech of the SAME frame T - CD. Same as the
  reference (Aloe rows, CD 3 there). No extra delay; at inference the speech codes of frame T must exist before the pose tower runs at T
  (the reference engine's job).
- Speech (podcast) rows and text rows never run `tower_p` (`_towers`: `kind == "pose"` only), so s2p cannot touch them structurally;
  proven below with real micros (no s_emb gradient, all terms bit-identical).
- s_emb is a ZERO-INIT CODE TABLE, not a table + zero-init projection (= ppm/pose-s2p README "The zero table is the output
  projection"): `S2PEmb.weight` [32x2048 or 2048, 2560] is summed straight into the tower input (`H2.code_feedback`, -1 -> 0), so
  step 0 is bit-identical AND every used row has a gradient at step 0 (tested below: non-zero rows == exactly the codes present).
  A random table + zero projection would give the table zero gradient at step 0.
- pose-CFG uncond row (silence codes): an INFERENCE-time rule. The 8,1,1 trainer has no pose-CFG / condition dropout (grep: none), so
  there is nothing to port in training.

## How the tests run
CPU only on the spares (sbatch, `nice -n 19`, 2 CPUs, small `--mem`; never the 16 production hosts, never slinky-9). The 8,1,1 trainer is
GPU-only, so `s2p_cpu.py` swaps the fused Triton / FA3 kernels for torch reference code, `autocast('cuda')` for `autocast('cpu')` and
`pin_memory` for identity IN THE TEST PROCESS ONLY; every compared variant runs the same swaps. Every micro goes through the trainer's own
`train_ppm.run_micro` with the 8,1,1 coefficients of `prod_811_fresh_append3.env` (pose micro 0 = backbone FULL, pose micro 1 = FROZEN,
speech micro 8, text micro 9): forward via the 8,1,1 bb/bridge path, both decoders' nanosteps (8 updates + their AdamW steps), one backward.
The tiny model is built by `model_ppm.build` (FSDP2 on a CPU world-1 gloo mesh, fp32 masters, bf16 compute, fp32 grad reduce), every
tensor seeded per name by the 811 init rules. Compared per micro: stats vector + loss (float.hex), sha256 of EVERY parameter gradient
(bytes, plus a value-level digest), the decoders' weights after their nanosteps, and the s_emb gradient (norm, non-zero rows vs the speech
codes present). Files: `s2p_cpu.py` (harness), `build_micros.py`, `test_cfg.py`, `run_variant.py`, `compare.py`, `test_resume.py`,
`real_weights.py`, `stage{1,2,3}.sbatch`; results in `out/`, logs in `logs/`.

## Real micros (build_micros.py -> /data/home/amartyavarma/ppm/pose5/s2p-test-micros, summary out/micros.json)
| row type | source (read-only) | N | facts |
|---|---|---|---|
| fresh pose | plan-1 segments of slinky-0 (fresh/scan) from the Weka copy, built by fresh_pose.FreshRows + pose_micro_ppm + packed (= FreshPoseDS) | 7,497 | ch A: 0 codes, present 0, ids all PAD; ch B: 7,382 Mimi positions = 7,382 pose positions = 7,382 pose labels; every pose label has B speech codes at the same T (0 without); label == pdepth[T,0] |
| podcast speech | evalsets/speech_s3/B-000.pkl (speech_micro path) | 7,502 | both channels coded (A 5,578, B 5,573 positions); no pdepth |
| Gemma text | data_ppm.TextDS on gemma-txtdup-v1 train pack 0 | 6,812 | no audio |
| chat text | chat_ppm.ChatDS split 2 pack 0 | 6,924 | no audio |
(The fresh pose row is built from real plan-1 segments with the production builder, but it is not one of the planned training rows:
those plans live node-local on the production hosts. Some plan-1 releases' parquets are missing from the Weka copy; only present ones used.)

## Results
### (a1) resolved cfg + real-size parameter table: PASS (job 30909, out/CMP_cfg.json)
- flag off: the copy's resolved CFG (115 keys, the json the trainer logs + saves) == code-811's byte for byte; COEF, RESUME_KEYS, NV, GA,
  GROUPS identical; the full-size PPM (64/36/28 layers, D33 link) has the same 2,150 parameters / shapes / LR groups.
- PPM_S2P=1: CFG adds exactly `s2p: full`, `s2p_from_step: 0` (every other key equal); params add exactly
  `tower_p.s_emb.weight [65536, 2560]`, group `tower_p` (+0.168B). PPM_S2P=cb0: `s2p: cb0`; `[2048, 2560]`, group `tower_p`.

### (a2) flag off byte-identical vs ORIGINAL code-811, tiny config, 5 real micros: PASS (job 30913, out/CMP_tiny_c128_orig_vs_copyoff.json)
- every micro (pose FULL, pose FROZEN, speech, Gemma, chat): stats vector + loss bit-identical; 0/92 parameter gradients differ (bytes);
  decoders after nanosteps identical. Noise floor: original vs original rerun also 0 differences (CMP_tiny_c128_orig_vs_orig.json).
- Run twice: lm_head CE chunk 128 (job 30913, out/*_c128_*; fits a 2.9 GB CPU slot) and the PRODUCTION chunk 1024 (job 30943,
  out/*_c1024_*). Every comparison below passes at both.

### (b-i) flag on, s_emb = 0 (init) vs flag off, tiny config: PASS at chunk 128 and 1024 (out/CMP_tiny_c{128,1024}_orig_vs_{full0,cb00}.json)
| micro | terms + loss | other 92 grads | decoders | s_emb grad (full) | s_emb grad (cb0) |
|---|---|---|---|---|---|
| pose FULL | bit-identical | 0 differ | identical | norm 0.0389, 53,284 rows non-zero = exactly the 53,284 distinct codes present | 0.00859, 1,520 rows = codes present |
| pose FROZEN | bit-identical | 0 differ | identical | 0.0389, same rows | 0.00859, same rows |
| speech | bit-identical | 0 differ | identical | None (no grad) | None |
| Gemma / chat | bit-identical | 0 differ | identical | None | None |

### optional: non-zero s_emb changes ONLY pose rows: PASS at chunk 128 and 1024 (out/CMP_tiny_c{128,1024}_orig_vs_{fullnz,cb0nz}.json)
- pose micros: pose cb0 CE sum 56,879.8 -> 56,957.9 (full) / 56,885.5 (cb0); speech cb0 + text CE sums on the same pose micro unchanged.
- speech / Gemma / chat micros: every term and all 92 gradients bit-identical, no s_emb gradient (the speech decoder too).

### (c) resume from a checkpoint WITHOUT s_emb, LR group, AdamW, strictness: PASS for PPM_S2P=1 (job 30943, out/resume_{A..L}.json) AND PPM_S2P=cb0 (job 30959, out/resume_cb0_{A..L}.json)
(cb0 suite: same phases with cb0 as the on-mode; C/K: only s_emb [2048, 2560] missing, s2p_from_step 17850 / 18022, 1,520 rows touched at
step 1 = the cb0 codes present; D/L strict; F = a PPM_S2P=1 config refusing the cb0 checkpoint; E/G/H/I refused as below.)
Real trainer save (`train_ppm.save` -> safety_ppm per-rank sharded file + rep.pt + digests) and real `s811_init` + `load()` of the tiny
FSDP2 model, synthetic checkpoints on node-local /scratch only.
| phase | what | result |
|---|---|---|
| A | ORIGINAL code-811, flag off: 1 step, save "dcp-17850" (no s_emb weight / AdamW state / cfg key) | PASS |
| B | copy flag off loads A strict | digest == A's: PASS |
| C | copy PPM_S2P=1 loads A: ONLY `tower_p.s_emb.weight` missing (allow_missing) | every other tensor + AdamW state == A (digest without s_emb); s_emb all 0, no AdamW state; group `tower_p`; LambdaLR groups unchanged (6); `s2p_from_step` 17850 |
| C (cont.) | 1 step (pose FULL + pose FROZEN + Gemma), save dcp-17851 | s_emb grad set by each pose micro, unchanged by the text micro; AdamW step: s_emb 1, other tower_p params 1 -> 2; 53,284 rows updated by ~lr each (max 3.0e-6 = lr 3.0e-6 at this warmup step); ckpt holds s_emb + its AdamW state, cfg `s2p: full`, `s2p_from_step: 17850` |
| D | copy PPM_S2P=1 loads C STRICT (s_emb pre-poisoned to 0.5) | digest == C's save; s_emb restored, has AdamW state; `s2p_from_step` carried 17850 |
| E | copy flag OFF loads C | REFUSED ("checkpoint has pose-s2p ... this config has PPM_S2P None") |
| F | copy PPM_S2P=cb0 loads C (full) | REFUSED (mode differs) |
| G | copy PPM_S2P=1 loads A with `tower_p.code_emb.weight` removed | REFUSED ("checkpoint parameter names differ") |
| H | copy PPM_S2P=1 loads A whose train_state claims s2p but lacks the tensor | REFUSED |
| I | ORIGINAL code-811 loads C | REFUSED (unknown `tower_p.s_emb.weight`): a rollback can never silently drop s_emb |
| J | LATER restart, leg 1: copy flag off loads A, 1 step, saves "dcp-18022" (no s_emb, no s2p keys) | PASS |
| K | leg 2: copy PPM_S2P=1 loads J | only s_emb missing; `s2p_from_step` = **18022**; s_emb 0, fresh AdamW (step 1) while tower_p's other params continue 2 -> 3; saves dcp-18023 with s_emb + state |
| L | copy PPM_S2P=1 loads K STRICT | `s2p_from_step` carried 18022; digest == K's |
AdamW / LR for the real resume: s_emb joins the EXISTING `tower_p` group (no new group, no LambdaLR entry, no own warmup); at 17850
tower_p's LambdaLR factor is 1.0, so s_emb's first update is Adam step 1 on a fresh state: |delta| = lr x |g| / (|g| + 1e-8) ~= 3e-4 per
element on every row with a gradient (the test shows exactly lr per touched element), then normal Adam; the rest of the model's AdamW
state continues exactly. `s2p_from_step` records the step s_emb's AdamW started (17850, or the leg-2 restart step).

### (b-ii) REAL widths + REAL 8,1,1 weights, real micros: PASS for full AND cb0 (out/CMP_real_orig_vs_{full0,cb00}.json)
ORIGINAL code-811 flag off (job 30947) vs copy PPM_S2P=1 (30980) / cb0 (30981), s_emb = 0, through run_micro with the production CE chunk:
| micro | loss (all three) | terms + loss | other 92 grads | decoders | s_emb grad full | s_emb grad cb0 |
|---|---|---|---|---|---|---|
| pose FULL | 37.994633 | bit-identical | 0 differ (90 set) | identical | norm 4.19e-3, 53,284 rows = codes present | 8.38e-4, 1,520 rows = codes present |
| pose FROZEN | 20.303928 | bit-identical | 0 differ (54 set) | identical | 4.19e-3, same rows | 8.37e-4, same rows |
| speech | 23.177967 | bit-identical | 0 differ (62 set) | identical | None | None |
| Gemma | 15.524290 | bit-identical | 0 differ (33 set) | identical | None | None |
| chat | 24.923191 | bit-identical | 0 differ (33 set) | identical | None | None |
CPU cost: ~11 min per pose micro, ~40 min per variant on 2 CPUs (no GPU).
Weights: ckpt-infer/ppm-v1/local-17641 (8,1,1 step 17641), reduced depth (backbone 2 layers, towers 2, decoders 1), real widths
(5120 / 2560 / 1024, vocab 151,936 / tower vocab 151,682). Cache (bf16, original-model tensors only, no s2p / g0 keys), read-only for D too:
`/data/home/amartyavarma/ppm/pose5/s2p-test-weights/real_w_2_2_1.pt` (12.5 GB; built by real_weights.py, job 30963; slinky-4 run 30947
uses an identical node-local copy /scratch/amartyavarma/pose5-s2p-test/real_w_2_2_1.pt on slinky-4). Jobs: ORIG off 30947 (slinky-4),
COPY full 30961 / cb0 30960 (slinky-18, 2 CPU / 38 GB each), compare 30950.

## Not covered / follow-ups (outside this port)
- Inference: code-811-s2p/infer_ppm.py (811's own engine) does not know s_emb: a DCP export with s_emb would be IGNORED there (extra key)
  and a local-shard export would KeyError. Samples / evals of an s2p checkpoint need the s2p engine (ppm/pose-s2p infer, being built).
- codep eval probe (pose rows, eval only): its speech_shuf / speech_sil variants change only the TRUNK's Mimi codes; s_emb reads the micro's
  real B codes (as every tower input does there), so codependence/* speech->pose under-states the new direct path. Unchanged by design
  (eval semantics); flag to the user if that metric is used to judge s2p.
- Test leftovers (tiny synthetic checkpoints, never real ones): /scratch/amartyavarma/pose5-s2p-test/ckpt on slinky-4 and slinky-18;
  weight caches /scratch/amartyavarma/pose5-s2p-test/real_w_2_2_1.pt (slinky-4) and /data/home/amartyavarma/ppm/pose5/s2p-test-weights/
  real_w_2_2_1.pt (12.5 GB, shared with D); micros /data/home/amartyavarma/ppm/pose5/s2p-test-micros. Not deleted (main decides).
