# PPM 8,1,1 stage (code-811)

User, 10-20 21:00 PT: *"prep the code for this 8,1,1 pose experiment. My goal is to focus almost all the gains of pose into the tower and light
pose and make sure the backbone doesn't forget the other tasks."* Prepared and tested 10-20. **LAUNCHED 10-20 23:13 PT as job 26391** (coordinator GO: switch now, link removed) from PPM's STOP_AFTER_SAVE
**dcp-16861** (PPM job 26325 stopped cleanly at 16861); its watchdog = job 26393 (PPM's watchdog 25617 cancelled).

`code-811/` = a copy of the production trainer `ppm/code/` (never modified) plus the 8,1,1 machinery, all behind `PPM_S811=1`. With
`PPM_S811` unset the trainer is the production trainer (tested: same losses, same eval, same checkpoint format).

| file | what |
|---|---|
| `train_ppm.py`, `model_ppm.py`, `s811.py` | the schedule (doc block "8,1,1 SCHEDULE" in train_ppm.py; pure helpers in s811.py) |
| `safety_ppm.py` | + `PROTECT_STEPS` (the switch checkpoint is never pruned); + test-only `PPM_TEST_LOAD_CPU` |
| `prod_811.env` | = `prod_ppm.env` verbatim + the 8,1,1 knobs (diff vs job 26271: `test811/CFG_DIFF_vs_26271.txt`) |
| `ppm811.sbatch`, `ppm811_watchdog.sh/.sbatch` | 16-node launch (job name `ppm811`) and its watchdog (own state files) |
| `LAUNCH_811.txt` | the launch command + switch procedure (STOP_AFTER_SAVE) + first-minutes checks |
| `test811/` | tests, test envs, logs in `ppm/logs/ppm811test/`, metrics in `ppm/runs-811test/` |

## Schedule, per optimizer step

`PPM_GA_SEQ=pose*8,speech,text` and `PPM_BB_MODE=full,frozen*7,full,full`:

- **8 pose micros** (one pose row per rank each). The pose tower and pose light decoder train on all 8. The light decoder does its usual
  `PPM_DEC_K=8` updates per micro over 100% of the micro's frames, giving **64 pose-light updates per step**. Frames are never
  subsampled: `light_frac = pose_light_frac = 1.0` is asserted, and per-step frame coverage is logged (`train/dec_p_frame_frac` = 1.0).
- **Pose micro 1 is backbone-full**: the production micro, with the trunk forward and backward.
- **Pose micros 2-8 are backbone-frozen**: the 32B trunk runs live every micro, forward only, under `no_grad`. There is no activation
  recompute, no trunk backward and no trunk reduce-scatter; the towers read `hn.detach()`. Nothing is cached.
- **1 speech + 1 text micro**, both backbone-full.
- Backbone gradients accumulate over the step's full micros. Then there is one clip and one AdamW step, with the same groups, LRs,
  clips, betas and constant schedule as PPM.

## Knobs and defaults

All knobs are set explicitly in `prod_811.env`.

| knob | default | meaning |
|---|---|---|
| `PPM_S811` | 1 | 8,1,1 on (unset/0 = production trainer) |
| `PPM_GA_SEQ` | `pose*8,speech,text` | micros per step (multiplier syntax; plain lists as before) |
| `PPM_BB_MODE` | `full,frozen*7,full,full` | per micro: `full` = gradients into the backbone, `frozen` = trunk forward only; text must be full |
| `PPM_BB_W_POSE/_SPEECH/_TEXT` | 1 / 2 / 1 | backbone task weights (normalisation below) |
| `PPM_TW_W_POSE/_SPEECH` | 1 / 2 | tower-side task weights |
| `PPM_W_NORM` | `task` | `task` (below) or `micro` (per-micro weights normalised per part) |
| `PPM_TOWER_CADENCE` | `step` | tower groups (tower_s, tower_p, decp_bb): `step` = accumulate all micros, step with the backbone; `micro` = clip + AdamW after every micro that fed them (tower_p 8 updates/step); the backbone always steps once |
| `PPM_DEC_K` | 8 | light-decoder updates per micro (frames always 100%) |
| `PPM_POSE_SPEECH_PARTS` | `all` | speech tower + speech light decoder on `all` pose micros (PPM's per-micro maths) or only on `full` ones |
| `PPM_POSE_EP2_SEED` / `PPM_SPEECH_EP2_SEED` | 20261021 | seeds of the pose / speech epochs after PPM's (epoch e uses seed + e - 2) |
| `PPM_PROTECT_SWITCH` | 1 | never delete the switch checkpoint `dcp-S0` |
| `PPM_811_CHECK_STEPS` / `_LOG_ROWS` / `_EVAL_REF` | 2 / 2 / PPM metrics glob | launch checks (below) |
| `PPM_MAX_STEPS` | 19063 | end of pose epoch 2 for the switch at 16861 (380 epoch-1 rows + 17,241 = 2,202 steps; general S0 + (34482 - S0) // 8) |
| `PPM_DECP_BB_OFF` | 1 | D33 backbone -> pose-decoder link REMOVED (ported from link-exp job 26303): module kept (checkpoint loads exactly), frozen, in no optimizer (AdamW state set aside, digest-checked), never computed in training (zb=None); asserted gradient-free every step; eval probe decp_linkon_* re-adds it; exports record alpha 0 |

### How the weights normalise (default `PPM_W_NORM=task`)

Each micro gets three coefficients, one per part:

- `c_bb` for the backbone (trunk, spec, proj, audio).
- `c_s` for the speech-tower side: tower_s plus the speech decoder's input gradient.
- `c_p` for the pose-tower side: tower_p, decp_bb, and the pose decoder's input gradient.

The formula is the same for each part:

`c = weight[task] / (number of micros of that task feeding the part) / Z`, with `Z = PPM_BB_W_POSE + PPM_BB_W_SPEECH + PPM_BB_W_TEXT`

- Z sums over the tasks present in the step.
- The backbone counts only full micros. The tower sides count every micro that runs them, full or frozen.
- Each task's weight is therefore its share of that part's gradient, split evenly over its micros.

**With the defaults every part gets exactly PPM's per-task coefficients** (PPM: each of pose, speech, speech, text × 1/4).

| part | pose | speech | text |
|---|---|---|---|
| backbone | 25% (the 1 full pose micro) | 50% | 25% |
| tower_s | 1/4 (averaged over the 8 pose micros) | 1/2 | — |
| tower_p | 1/4 (averaged over the 8 pose micros) | — | — |

The towers see an 8× pose batch. To tune the backbone (e.g. to protect it further), change `PPM_BB_W_*`; the towers are unaffected.

A full micro's backward roots are:

- `c_bb × text CE`
- `+ c_s × cb0`, `+ c_p × pose cb0`
- `+` the decoder input gradients × `c_s` / `c_p`

The towers read `hn` through `GradScale(c_bb / c_side)`, so the trunk receives exactly `c_bb` × the micro's whole gradient. Tested both on
a toy graph and end to end.

## Unchanged from PPM

Losses, loss weights (text ×5 etc.), LRs, clips, AdamW (betas, eps, fused), constant LR, warmups (done), decoder nanosteps, evals and the
dependence probes all stay as in PPM. So do checkpoint content, format and digests, hourly node-local saves (keep 2), 6-hourly Weka
exports, and the control files.

- **Run base:** stays `ppm-v1`, so the checkpoint dir, exports, the HF repo chain and the live page chain all continue.
- **HF backup:** `setup-logs/hfb/ppm_job.sh` now also recognises job `ppm811` / `code-811/ppm811.sbatch`, and `snap_code.sh` snapshots
  `code-811` for it. Backups: `.bak-pre-811`.
- **Watchdog:** `ppm811_watchdog.sh` is PPM's watchdog with the 8,1,1 code, env, sbatch, job name and target, and its own state files.
- **Live page:** `stage3_ppm.html` shows the stage as a continuation of PPM main. Rows/epochs are piecewise per schedule, with a dashed
  "8,1,1 from step S0" marker on every chart. It is a no-op until the first 8,1,1 log exists.

## Data continuation (proposal, implemented)

PPM consumed, by step S0: 1 pose row per step, 2 speech rows per step, and 1 text pack per step.

- **Pose:** first the rest of PPM's epoch-1 order. From 17366 there is none: PPM finished epoch 1 at 17241 and re-read 125 rows. Then
  epoch 2, 3, … as **fresh node permutations** (`pose_s3q.plan`, seed `PPM_POSE_EP2_SEED + e - 2`; PPM used 20261011), at 8 rows per step.
  - From 17366, epoch 2 is 17,241 rows/rank = 2,155 steps (to 19521).
- **Speech:** first the rest of PPM's D2 stream. From 17366 that is only ~1-27 rows/rank: PPM's speech epoch also ends at 17366. Then
  epochs of the same rows, holdout episodes excluded, permuted per rank with `PPM_SPEECH_EP2_SEED + e - 2`, at 1 row per step.
- **Text:** continues PPM's per-source pack order by global step (Gemma world-strided packs and chat packs, 50/50 by step hash). There is
  1 text micro per step in both schedules.
- **Later resumes:** the switch step and PPM's GA are stored in every 8,1,1 checkpoint (`cfg.s811`), so a resume rebuilds the identical
  streams. The epoch seeds must match.

## Resume

`PPM_RESUME_FROM=auto` picks the newest dcp COMPLETE on all 16 hosts.

- **From a PPM checkpoint (the switch):** weights, the main AdamW, both decoder AdamWs, the schedule and the decoder counters are
  restored. The per-rank digests are asserted exactly as in PPM. `ga_seq` is the only resume key allowed to differ.
- **From an 8,1,1 checkpoint:** everything data-defining must match. Weights, mode and cadence are resume-changeable and are logged.
- **Resolved-cfg diff vs job 26271:** `ga_seq`, `max_steps`, `run_tag` and the `s811` block (+ `s811_*` launch-check keys). Nothing else.

## Tests

The 16 nodes were busy, so the tests ran on 2 GPUs (slinky-18 GPUs 6-7) at reduced depth: 2 trunk, 2 tower and 2 decoder layers.

**CPU**

- `test811/test_s811_cpu.py` (26/26):
  - schedule parsing;
  - coefficients (defaults = PPM's exactly; PPM GA in 8,1,1 mode = 1/4 everywhere);
  - streams (epoch-1 rest → epoch 2, seeded speech epochs, consumed-row accounting, determinism);
  - the GradScale bridge on a toy graph (trunk grad = c_bb × all, tower grads = c_side × theirs).
- `test_protect_cpu.py`: prune keeps the switch save.
- `test_holdout_cpu.py`: the speech holdout mask equals the D2 one.

**GPU** (`logs/ppm811test/t*.log`, `test811/compare_runs.py`)

- **Grad isolation (tA, R2, R3, tB, tC):** every frozen micro leaves the backbone gradients **bit-identical** (`torch.equal` on every rank,
  asserted). Every full micro changes them, and every micro's tower side accumulates.
- **Light (tA, R2):** dec_p does **64 updates per step**, frames 58,291/58,291 (100%). dec_s does 72 updates (9 micros × 8), 100%.
  AdamW counters move +1 per step.
- **Cadence micro (tB):** tower_s +9, tower_p +8, decp_bb +8, trunk +1 per step.
- **Speech parts `full` (tC):** dec_s 16 updates per step, frozen pose micro 0.41 s vs 0.60 s.
- **Resume (R1 → R4/R5/R2, R2 → R3):**
  - R1 is PPM mode, 3 steps, saved.
  - R4 (PPM resume), R5 (8,1,1 mode with PPM's GA, all full) and R2 (8,1,1 switch at 3) all restore with digests asserted.
  - R2 → R3: R2 saved dcp-5 with SAVE_NOW and continued; R3 resumed from dcp-5. The rows of steps 6-7 are identical on both ranks.
- **Step-0 losses vs PPM at the same weights:**
  - The start eval of R4, R5 and R2 matches R1's eval at step 3: **77/77 metrics exact**. The trainer logs this check itself at launch
    (`[811] START EVAL vs PPM's eval`).
  - The first-step train losses of R5 equal R4's (33/37 exact; the rest ≤ 5e-5, speech light CE, from the decoder's non-deterministic
    backward). Step 5 is within 7e-4.

## Timing (16 nodes)

| micro | seconds | source |
|---|---|---|
| pose frozen | **6.46** | 16-node production profile `ppm-19013.log` (same config): fwd 1.55 + dec 3.97 + tower bwd 0.94. The decomposition matched the 2-GPU test (0.596 = 0.14 + 0.37 + 0.08). There the no-grad forward was faster, so 6.46 is conservative (5.9-6.5) |
| light per-update overhead | **~0.03** | `test811/bench_dec_update.py`, full 0.575B decoder: clip + AdamW 9.2 ms + 2.3 GB all-reduce (8.3 ms on 2 GPUs; ~10-30 ms at 16 nodes). It only matters if `PPM_DEC_K` changes; the micro times already include K = 8 |

`steptime.py --step 'pose@full,pose*7@frozen,speech@full,text@full'` gives:

- **8,1,1:** 66.5 s/step (69.6 s with eval/save), ~52 steps/h.
  - Pose epoch 2 (2,155 steps) takes **~42 h**.
  - Rows/h: pose 53k (PPM 15.5k), speech 6.6k (PPM 30.9k), text 6.6k (PPM 15.5k).
  - Compute: backbone 33%, towers 18%, light 49% (PPM: 61 / 14 / 24).
- **PPM today:** 28.5 s/step.
- **With `PPM_POSE_SPEECH_PARTS=full`:** ~48.6 s/step.

## Open decisions for the user

1. **Backbone task weights** `PPM_BB_W_POSE/_SPEECH/_TEXT`. The default 1/2/1 is PPM's backbone mix. Lower `PPM_BB_W_POSE` (e.g. 0.5)
   to push the backbone further away from pose.
2. **Pose epoch-2 seed** `PPM_POSE_EP2_SEED`, default 20261021 (fresh vs PPM's 20261011). The speech epoch-2 seed is the same.
3. Speech tower + speech light on the 7 frozen pose micros, `PPM_POSE_SPEECH_PARTS` (`all` = PPM per-micro maths, default):
   - With `all`, the speech light decoder's updates become 8:1 pose-row : podcast (PPM 1:2). Its gradient mix stays 1:2 through the
     weights.
   - `full` is ~27% faster per step but trains the speech parts on fewer pose rows. That is a recipe change and needs your OK.
4. **Run length** `PPM_MAX_STEPS`. Default 19521 = the end of pose epoch 2 from dcp-17366, ~42 h.
