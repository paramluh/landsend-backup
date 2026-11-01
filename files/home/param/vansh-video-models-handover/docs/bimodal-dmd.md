# Bimodal DMD

`learn_dmd_bimodal.py` distills the bimodal model with velocity DMD and a flow-matching critic. The same checkpoint initializes a trainable student, a frozen teacher and a trainable critic: the 850-step local-Mimi audio fine-tune of `aloe-bimodal-studio` step 10606 EMA, at `/mnt/paramfs/param/outputs/mimi-local-ft/aloe-bimodal-studio-mimi-local-audio-1k/run_1/step850`.

| | |
|---|---|
| Config | `experiments/distillation/aloe-studio-bimodal-dmd.toml` |
| Code | `distillation/bimodal_net.py` (network), `distillation/bimodal_dmd.py` (rollout, scoring, losses) |
| Data | `aloe-studio-no-blackbands-high-quality` (16,325 chunks, 1,360 h, studio-app and studio-studio) |
| Conditioning | electron-v2 pose, `mimi_local.safetensors` codes |
| W&B | project `param-distillation` |

## One outer step

Each row has the layout `sink | GT 1..3 | targets 4..7`. Every row is a full window: there are no startup windows, pads or history dropout. One outer step on each of the 40 ranks:

1. **Critic pool.** The student rolls out 16 rows without gradients. It returns the frames it stabilised to t = 0.2 and the frames it resolved to t = 0.
2. **Critic updates.** Four updates of 4 rows each (effective batch 160). Each is flow matching on the student's samples:
   - the stabilising window, weighted 0.8;
   - the four resolving windows, weighted 0.2.

   With `critic_combined`, both go through one 17-frame pass with a shared sink.
3. **Student update.** One update of 4 one-row microbatches (effective batch 160):
   - **2 stabiliser-scored rows.** The student rolls out only to t = 0.2; the resolver is never run. The 4 frames are re-noised into the 4 stabilising bands and scored in `[sink | GT 1..3 at t = 0.2 (fresh noise) | 4 present]`.
   - **2 resolver-scored rows.** The student rolls out fully. Each frame gets its own `t ~ U(0.02, 0.4)` and is scored in its own resolving window `[sink | frames f-2, f-1 clean | f]`. Those clean frames are GT for the seeds and the detached generated frames after that. The four windows share one pass.

   The teacher and critic score without gradients. The rollout keeps gradients through every pass, including through the cached K/V that later frames read.

The critic gets 150 warmup outer rounds before the first student update. Both optimizers use AdamW: LR 2e-6, betas (0.9, 0.99), weight decay 1e-4, clipping 2. LR warms up over 100 student / 200 critic updates. The student EMA has a 200-update horizon.

## Noising a t0 = 0.2 sample, and the DMD and FM targets

The generator output `y` lives at level `t0`: 0.2 for stabilised frames, 0 for resolved ones. Scoring and the critic re-noise it with a fresh draw:

```text
x_t = a y + s eps,    a = (1 - t) / (1 - t0),    s = sqrt(t^2 - (t0 a)^2)
v*  = a' y + s' eps,  a' = -1 / (1 - t0),        s' = (t + t0^2 a / (1 - t0)) / s
```

- **Correct marginal.** If `y = (1 - t0) x0 + t0 n`, then `x_t = (1 - t) x0 + (t0 a n + s eps)`. The noise has variance `t^2`, so `x_t` has the rectified-flow marginal at `t`.
- **FM target.** `v*` is the path velocity. Its regression target equals FM's: `E[v* | x_t] = E[eps_total - x0 | x_t]`.
- **t0 = 0.** `v* = eps - y` exactly.
- **Critic loss.** `mean |vF - v*|^2`, weighted 0.8 for stabilising and 0.2 for resolving.
- **DMD.** The universal velocity DMD carries over unchanged:
  - `g = (vT - vF) / max(mean|vT - v*|, 1e-8)` per frame;
  - `L = 0.5 |y - sg(y - g)|^2`.

  For `t0 = 0.2` this equals DMD's normalized gradient in `x_t0` space, `(E_F[y|x_t] - E_T[y|x_t]) / mean|y - E_T[y|x_t]|`.
- **Scoring times.**
  - Stabilising bands are those of `(0.2, 1]`, intersected with `[0.22, 0.98]`. Near `t = 0.2` the target's `s'` diverges, and `t0 + 0.02` mirrors the universal `0.02`.
  - Resolving uses `t ~ U(0.02, 0.4)`.

**Check.** Run through the trainer's own `noised()` on bimodal non-Gaussian data, with 20M samples per `t`:
- At `t0 = 0` the path is exactly rectified flow.
- At `t0 = 0.2` and `t ∈ {0.22, 0.5, 0.8, 0.98}`:
  - The marginal of `x_t` matches `(1 - t) x0 + t eps`, with quantile gaps of 0.002–0.005 (sampling noise).
  - `E[v* | x_t]` matches `E[eps - x0 | x_t]` in 16 bins. Mean |z| is 0.66–0.89 and max |z| is 2.5, as expected for equal means.

**Code checks.** These run on CPU with a tiny random bimodal net (4 layers, audio blocks at layers 1 and 3), using exact SDPA attention in place of FlashAttention:
- The padded static `UniformRollout` matches the per-pass trimmed `Rollout`, which earlier matched the production cached sampler and `ResolverNet` at the teacher's 10-step schedule.
  - Stabilised and resolved outputs agree to a relative L2 of 1.5e-6.
  - All 92 parameter gradients agree to within 3.6e-3 (bf16 rounding of q/k/v), with none missing. This holds for both the stabilise-only and the full rollout.
- The packed resolving pass equals four separately built `[sink | f-2, f-1 | f]` windows (7.8e-8).
- The DMD loss's autograd gradient equals `(vT - vF) / normalizer`, averaged over the mean (6.1e-8).

## Metrics

Each top-level W&B folder holds a few series:

| Folder | Contents |
|---|---|
| `dmd_stabilizer/`, `dmd_resolver/` | loss, normalizer, \|vT - vF\| |
| `dmd_*_frame/` | per-frame DMD loss for frames 0..3 |
| `dmd_*_tbucket_loss/`, `dmd_*_tbucket_normalizer/` | DMD loss and normalizer by t bucket (stabilizer: 0.1 buckets of (0.2, 1]; resolver: 0.02, 0.1, 0.2, 0.3, 0.4) |
| `fm_stabilizer/`, `fm_resolver/`, `fm_*_frame/`, `fm_*_tbucket/` | critic flow-matching losses |
| `student/`, `critic/` | loss, grad norm, LR |
| `seconds/` | phase times, max over ranks |
| `seconds_spread/` | phase times, median rank and slowest rank |
| `memory/` | peak allocated GiB |

## Implementation

**Weights and gradients.**
- Weights are replicated in bf16. Student and critic keep ZeRO-1 fp32 masters with AdamW (`utils.harness.Zero1FP32MasterAdamW`).
- Gradients are views into flats laid out like the parameters and reduce-scattered to their owners.
- FSDP's per-layer all-gathers would dominate batch-1 rollouts.
- The frozen teacher's 30 blocks are kept as 1/8 shards per node. Each block is all-gathered over NVLink just before it runs, which saves about 12 GB per GPU.

**One functional forward** (`BimodalNet.forward_pass`).
- It covers the rollout, the stabilising window and the packed resolving windows.
- Frames are listed with their slot, time and validity. Each frame's keys are listed in the plan.
- A pass returns K/V only for the frames later passes read.

**Rollout** (`UniformRollout`). It runs the fused schedule from `bimodal/student_rollout.py` with one static layout:
- every pass is `[4 stabilising | resolving | stabilising-history commit | resolving-history commit]`;
- empty slots are pad frames;
- 2 Euler steps per band and 2 resolving steps (from t = 0.4).

**Checkpointing** (`_BlockCheckpoint`).
- Each block's forward runs without autograd. Backward re-runs the block through the same compiled code and backpropagates it on its own.
- `torch.utils.checkpoint` recomputes with Dynamo disabled, which saves different tensors than the compiled forward and fails its metadata check.
- Plain autograd sums every weight's gradient across all 16 rollout passes before accumulating it, a second set of about 13 GB. Accumulating per block avoids that.

**Compilation** (`compile_mode = 'block'`). Whole blocks are compiled at static shapes: one graph per layout × batch size × gradient mode × layer type. The rollout uses a single layout.

**Garbage collection.** `gc.freeze()` runs after setup and during the first steps of each phase. Automatic collection is then disabled, and every rank collects at the same step boundary every 50 steps. Left on, a collection on one rank mid-backward stalled all 40 at the gradient reduce-scatter.

## Benchmark (40 H100s, 5 nodes, real data, October 14, 2026)

Steady-state seconds per outer step (pool + 4 critic updates + 1 student update, each at effective batch 160). The compile time is to the first step.

| Rollout layout / compilation | Compile | Graphs | s/step | Pool | Critic | Student | Peak GiB |
|---|---|---|---|---|---|---|---|
| Per-pass trimmed layouts, whole blocks static | ~21 min | 904 cache entries | 18.5 (outliers 23.6, 47.7: late compiles) | 3.1 | 4.2 | 11.3 | 73 |
| Per-pass trimmed layouts, dynamic-shape compiled parts | ~4.5 min | ~20 | 23.7 | 3.7 | 5.1 | 15.0 | 67 |
| **One padded layout, whole blocks static** | **~5 min** | **few** | **18.5** | 5.3 | 4.2 | 9.3 | 69 |
| + combined critic pass + GC freeze/disable (live run) | ~5 min | few | **18.1** | 5.3 | 3.9 | 9.1 | 69 |

**With `pool_rollout = 'fused'` (current).** The no-grad batch-16 pool uses the trimmed per-pass layouts: they are inference-only, about 28 extra graphs and ~2 extra minutes of compile. The padded student rollout is unchanged. Live run `s81uabdw`:
- warmup rounds 6.87 s (pool 3.15 s, critic 3.87 s);
- outer steps 15.86–15.98 s (pool 3.13 s, critic 3.83 s, student 9.0–9.1 s);
- peak 69.2 GiB.

One epoch is about 42 h. The rows below are the previous run with the padded pool.

Final settings only, from the live production run (wandb `5hdqti6g`, all 40 GPUs, real data):

| Phase | n | Median s | Mean s | Min s | Max s |
|---|---|---|---|---|---|
| Critic warmup round (pool + 4 critic updates) | 149 | 9.06 | 9.06 | 8.99 | 9.08 |
| Outer step (pool + 4 critic + student) | 18 | 18.19 | 18.27 | 18.06 | 19.59 |
| Pool | 18 | 5.32 | 5.32 | 5.28 | 5.34 |
| 4 critic updates | 18 | 3.96 | 3.96 | 3.91 | 3.99 |
| Student update | 18 | 9.11 | 9.21 | 9.03 | 10.53 |
| Gradient reduce-scatter + optimizer + all-gather | 18 | 0.47 | 0.54 | 0.34 | 1.86 |

Peak memory is 69.2 GiB.

How to read this:
- **Padding's cost.** It only costs time in the compute-bound batch-16 pool. Batch-1 student passes are bound by weight reads, so they are no slower with 7 frames than with 4–5.
- **Batch-2 student microbatches.** Same objective, but out of memory with the padded layout.
- **Rejected designs.**
  - Per-pass layouts in compiled code gave no speedup over padding and cost 20+ minutes of compilation.
  - Compiling only the dynamic-shape parts costs 5 s per step: the layout-specific attention runs eagerly.
- **Before the GC fix.** The same settings ran 21–25 s per step, because the gradient sync took 0.7–2 s instead of 0.35–0.5 s.
- **Projection.** One pass over the 7.64M rows (9,550 outer steps at 800 rows per step) takes about 48 h at 18.1 s per step. That excludes the 150 warmup rounds (9.1 s each) and checkpoint I/O.
- **Comparison.** The universal DMD benchmark (H4/P5, 18–26 calls per rollout) measured 22–26 s per outer step.

## Launch

```bash
bash run_dmd_bimodal.sh "192.168.0.4 192.168.0.9 192.168.0.7 192.168.0.6 192.168.0.8" \
    experiments/distillation/aloe-studio-bimodal-dmd.toml
```

- **Code location.** Run from a code snapshot readable by every node; the current one is `/mnt/paramfs/param/outputs/bimodal-dmd/code`.
- **Other arguments.** `--set key=value` overrides the config.
- **Benchmarking.**
  - `--benchmark N` times N outer steps with no warmup, saves or W&B.
  - `--variants "k=v,k=v|k=v"` with `--variant-steps` cycles settings within one launch.
- **Checkpoints.** Every 200 steps, the run writes `student.safetensors`, `critic.safetensors`, `model_ema.safetensors` and per-rank optimizer/EMA state.
- **Stopping.** `touch RUN_DIR/save_q` saves and quits.

## Richmond: local data, local disks, and reusable critic warmup

`experiments/distillation/aloe-studio-stabilizer-dmd-richmond.toml` uses the original studio `shard.csv` on each node,
with pose-v2 restored into `/home/param/datasets/aloe-studio-v2`. The initial teacher and frozen Mimi frontend live
under `/home/param/richmond-assets`. Outputs go to `/mnt/paramfs/param/outputs/bimodal-dmd`; on Richmond this path is
local to the rank-0 node and must be archived through the migration bridge. It is not a shared mount.

The config scores four stabilizer student rows, trains the critic with stabilizer FM only (`fm_weight_res = 0`),
and disables EMA (`ema_decay_steps = 0`). A zero resolver FM weight now also skips resolver generation in the
critic pool and resolver critic passes. Evaluation uses the separately frozen teacher resolver; the updated
student checkpoint is not intended to be used as the resolver. The static student layout still contains its
old padding slots; removing those for speed is separate work.

With `node_sharded = true`, the eight local ranks partition their node's CSV, and every rank uses the minimum
full-window epoch length across nodes. Sampling repeats deterministic shuffled epochs to reach the configured
outer-step budget. Shared-data configurations still partition using global rank and world size.

`run_dmd_bimodal.sh` accepts Richmond hostnames or the Richmond IPs in `nodes.sh`. It maps the latter to hostnames
and selects `ibs14`, avoiding the duplicated NAT address on `ens1`. It sources the checkout's private `.env`,
streams each node's output to the terminal, and reports failure if any SSH worker fails. Example (not launched):

```bash
bash run_dmd_bimodal.sh 'richmond-9 richmond-8 richmond-10 richmond-11' \
    experiments/distillation/aloe-studio-stabilizer-dmd-richmond.toml
```

For local disks, `distillation/checkpoints.py` uses `utils.saver`'s existing CPU/Gloo transfers to collect all
optimizer shards on rank 0. Only rank 0 needs the checkpoint directory. An atomic `checkpoint.complete.json`
records a generation UUID and expected file sizes after every shard and the metadata finish writing. Its UUID
matches `metadata.json`. Overwriting or pruning a checkpoint invalidates the marker first; pruning old optimizer
shards produces a fresh marker and marks that checkpoint as weights-only. EMA, when enabled in another config,
is merged over the same transport. A disk-space check runs before saving; checkpoints are not deleted in advance
to make room. Prefer a rank-0 node with ample free disk, such as Richmond 9 or 12, and archive completed saves.

- `save_warmed_critic = true` saves `critic-warmup/` immediately after the last critic-only warmup round, before
  any student update. Its optimizer shards remain until a newer training checkpoint completes; then the
  latest-checkpoint-only policy prunes them along with older student optimizer shards. Warmup weights remain.
- `critic_warmup_only = true` ends after those warmup rounds. Use this with `save_warmed_critic = true`.
- `critic_resume_from = '/path/to/critic-warmup'` restores critic weights through its fp32 masters, Adam state,
  learning-rate scheduler, and critic update count. Student weights and optimizer stay freshly initialized,
  and critic warmup is skipped. The initial checkpoint SHA256, rollout schedule, scoring conditions, audio
  config, latent geometry, data corpus, and world size must match. Corpus identity uses the union of clip paths,
  lengths, and bad-audio flags, allowing the same corpus to be repartitioned; different subsets are rejected.
  Overlapping node shards are rejected before model loading. The saved critic optimizer settings are retained.
- `resume_from = '/path/to/step-000200'` restores both optimizers, schedulers, counters, RNG, and optional EMA.
  Data CSV hashes, batching, training settings, and world size must match. The new run continues the saved data
  cursor and step counters; it creates a new W&B run.

Both resume modes require the new completion marker, `metadata.json`, and rank-state format. Historical checkpoints without that
metadata are not accepted automatically. ZeRO-1 optimizer resharding between 32 and 40 GPUs is not implemented;
keep the same world size when reusing a warmed critic, or prepare one warmup per world size.

Validation: `BIMODAL_TEST_OUTPUT=/path/outside/repo python -m distillation.test_infra` runs four CPU/Gloo ranks to
check unequal local data shards, disjoint local rank sampling, rank-0-only checkpoint round trips with real
ZeRO-1 Adam state, critic-only restoration, RNG restoration, optional EMA, and compatibility rejection.

## Paired stabilizer runs: full weights and LoRA

`experiments/distillation/stabilizer-plan1.toml` uses the completed pose-v3 teacher at
`/home/param/richmond-assets/teacher-pose-v3-step200/model.safetensors`, the node-local
`aloe-studio-10shards` CSVs, and `pose_latents_electron_v3.safetensors`. It keeps four critic updates of four rows,
four one-row stabilizer student microbatches, the `[2, 2, 2, 2]` band schedule, a fused no-grad pool, a uniform
grad-tracked rollout, whole-block compilation and block checkpointing. All four generated frames and their
cached K/V stay differentiable. The resolver loss is zero and no resolved output is generated.

The budget is **150 critic warmup rounds plus 1,050 student outer steps**, ending at logical step 1,200.
W&B and checkpoint directory names use `warmup_step + outer_step`. The warmed critic is saved before the first
student update; the paired student checkpoints are logical steps **350, 550, 750, 950, 1150 and 1200**.
The last checkpoint is saved even though 1,050 is not a multiple of the 200-student-step save interval.

Both speedy networks use the same `lora` block (`rank`, `alpha`, `targets`). Only the matched linear adapters
are trainable; base weights are frozen. The initial rank-32 benchmark targets transformer attention q/k/v/o,
both FFN projections, and audio-adapter q/kv/output projections. A dense warmed critic is loaded before adapter
injection with `critic_resume_optimizer = 'reset'`: the new adapter tensors start with fresh Adam state.
The full dense warmup optimizer was saved and used for exact dense restoration during learning-rate calibration;
its older optimizer shards were then pruned under the latest-checkpoint-only retention policy. This transition
does not reinterpret dense moments as adapter moments. The speedy student and critic use a fresh 50-outer-step
LR ramp (50 student updates and 200 critic updates at four critic updates per outer step) and begin logging at logical step 151
without repeating the critic warmup.

`critic_resume_allow_data_change = true` explicitly permits the warmed critic to move between different node-local
training subsets. If the two five-node groups use nonoverlapping subsets, corpus differences are a confound in
the final model comparison. Paired evaluation fixes the evaluation inputs, but does not remove that training-data
difference. Teacher, conditioning and rollout compatibility checks still apply.

### Timing the speed target

The historical **15.86–15.98 s/step** above is the old mixed stabilizer/resolver run. The measured new full-weight
stabilizer baseline is **13.316 s/step** on 40 H100s (p10–p90: 13.298–13.337 s), so the 3× target is **4.439 s/step**.
The initial completed sweeps are below; later candidates remain in the benchmark artifacts.

| Case | Median s/step | Speedup | Peak GiB/GPU | Student / critic batch |
| --- | ---: | ---: | ---: | ---: |
| Full-weight baseline | 13.316 | 1.00× | 66.6 | 160 / 160 |
| Dense, small batches | 4.214 | 3.16× | 65.3 | 40 / 40 |
| LoRA32, small batches | 4.157 | 3.20× | 38.6 | 40 / 40 |
| LoRA32, compact layout | 4.071 | 3.27× | 38.2 | 40 / 40 |
| LoRA32, compact, skip six student checkpoint blocks | 3.625 | 3.67× | 46.7 | 40 / 40 |
| LoRA32, compact, batch 80 | 4.669 | 2.85× | 39.9 | 80 / 80 |
| LoRA32, compact, student 160 / critic 80 | 5.850 | 2.28× | 49.8 | 160 / 80 |

The selected first paired attempt keeps both global batches at **160**. The earlier reduced-batch cases above are exploratory measurements, not the final batch choice. The final memory search compares no-checkpoint microbatches 4→2→1 (accumulating to four rows/rank) against four rows together with progressively more checkpointing. The no-grad pool uses all 16 rows/rank together.

The compiled memory probes used **one node / eight H100s**, three full training steps and one skipped warmup
step. Every case retained four total student rows per rank, four critic rows per update and four critic updates;
the critic had no checkpointing. These are memory diagnostics with global batch 32, **not official 40-GPU timings**:

| Student microbatch | Student activation retention | Outcome | Warm s/step | Peak allocated GiB/GPU |
| ---: | --- | --- | ---: | ---: |
| 4 and 2 | All blocks, no checkpointing | OOM during student forward | — | — |
| 1 | All blocks, no checkpointing | Three steps passed | 9.410 | 67.59 |
| 4 | Retain 15 specified non-audio blocks | OOM during student forward | — | — |
| 4 | Retain blocks `0,1,2,3,5,6,8`; checkpoint the other 23 | Three steps passed | 6.578 | 73.97 |

The search stopped at the first fit in each branch; it did not refine upward afterward. The selected candidate
keeps all four student rows together, uses the fused full-gradient rollout, rank-32/alpha-32 LoRA in both
networks, BF16 frozen adapter-base weights and fused LoRA Q/K/V. The one-node timings exclude compilation and
do not establish the 3× target. Source records are under `benchmarks/frontier-no-checkpoint/` and
`benchmarks/frontier-batch4/`; the authoritative 40-GPU check is recorded separately under `benchmarks/fixed160-final/`.

The final five-node check of `b160-retain7-compiled` completed five steps, excluding the first two
from timing: median **6.5073 s**, p10–p90 **6.5014–6.5082 s**, peak **73.8285 GiB/GPU**.
Both global batches remain 160, giving **2.046×** the 13.3159 s baseline throughput and a
113.9-minute projection for 1,050 student updates before compilation/checkpoint I/O.
This does **not** meet the original 3× target. It uses trimmed full-gradient rollouts, four
student rows together, no critic checkpointing, BF16 frozen LoRA target weights, fused adapter
QKV, and ordinary block compilation. The CUDA-graph variant passed small-model numerical checks
and one full-model step, but was stopped for startup/capture time; it has no valid steady-state
timing and is not selected for training.

Only the latest completed checkpoint per run retains optimizer state. Older matching-step
weights stay available for evaluation. The archived warmup optimizer is removed after learning-rate
calibration; its weights remain a valid explicit `critic_resume_optimizer = 'reset'` bootstrap.
Full optimizer resumes continue to reject checkpoints whose optimizer shards have been pruned.

Critic batch is per update; all rows retain four critic updates. Most of the initial speedup comes from processing
fewer rows per logical step. These measurements establish throughput, not comparable trained quality. Their
source JSONs are under `/mnt/paramfs/param/outputs/stabilizer-dmd-20261015/benchmarks`; the evaluation directory's
`summarize_benchmarks.py` regenerates the combined table, quantiles, phase timings and 1,050-step projections.

`distillation/benchmark.py` runs each case in a fresh trainer process because batching, model construction and
LoRA cannot safely change inside the trainer's existing `--variants` loop. It streams all output, saves each log
and JSON summary, skips four initial steps by default, and reports median/p10/p90, phase times, peak allocated
memory, effective batch sizes and the 1,050-student-step projection. Its observed line-to-line cycle timing also
includes data loading and reporting; the trainer's `seconds/wall` excludes those. Phase subtimers inside student
microbatches are means per microbatch, not totals across all student rows. Projections exclude warmup, checkpoint
I/O and first-launch compilation. Results capture the local launch checkout's trainer, network, rollout, LoRA
and checkpoint source SHA256s before starting the subprocess; this does not independently verify remote files.

Use the **dense plan-1 config as the base**, since missing `lora` is how dense training is selected:

```bash
python3 -m distillation.benchmark run \
  --nodes 'node-a node-b node-c node-d node-e' \
  --config experiments/distillation/stabilizer-plan1.toml \
  --matrix experiments/distillation/stabilizer-speed-matrix.toml \
  --case dense-baseline --case dense-small --case lora32-small \
  --steps 12 --skip 4 --output /path/to/benchmark-output
```

For an aggressive ordered memory search, `run --memory-frontier --matrix MATRIX` tries cases in matrix order, continues only after an explicit CUDA out-of-memory failure, and stops at the first fitting case. Other failures abort. Profiled steps are excluded from timing summaries; non-finite loss/gradient results are rejected.

The matrix changes `critic_rows`, `student_kinds`, `pool_micro_batch`, `student_micro_rows`, checkpoint policy,
rollout layout and the LoRA settings. Dense-small and LoRA32-small use one critic row per update, one student row and a four-row
pool, retaining four critic updates. The optional `lora32-no-checkpoint` case removes recomputation and needs
its memory usage measured before adoption. `rollout = 'uniform_stab'` is another opt-in exact-layout candidate:
five static slots `[four stabilizing | history commit]` remove the two resolver slots without truncating the
four-frame gradient path (`lora32-compact`). `lora32-compact-skip6` additionally stores activations for the first
six student blocks and all critic blocks, retaining recomputation for later student blocks. These are performance
candidates, not evidence of equivalent trained quality.

The existing variable-size `rollout = 'fused'` is also valid with gradients: with two steps per band it evaluates
50 frames across 14 passes, versus 70 for `uniform_stab`, retaining all 44 real velocity predictions. It omits
14 padded frames and reuses six history commits instead of recomputing them on the second substep. Both layouts
add two one-frame prefills. `python -m distillation.test_trimmed_rollout` checks outputs, all parameter/input
gradients, dense/nonzero LoRA, fused LoRA Q/K/V, block/call checkpointing and selectively skipped blocks at batch
four. A last-frame-only loss still reaches noise for all four generated frames. With fp32 attention intermediates,
the complete CPU matrix agrees to at most 1.20e-7 absolute error; native BF16 intermediates produce small dense
gradient accumulation-order differences (up to 6.96e-5 absolute and about 0.24% tensor-relative L2 in these tests).
The variable layouts may incur additional compilation or reduce GPU kernel efficiency, so frame savings alone
do not establish a wall-clock gain.

To compare a later case against an earlier run, add `--reference /path/to/dense-baseline.json`. For existing
trainer logs, `report --config CONFIG --log LOG --world-size 40 --output baseline.json` creates a reference.
The report records batch reductions separately from step-time speedup. It rejects different world sizes,
teachers, schedules or objectives. A later change in training data must still be accounted for independently.

### Identical evaluation windows at each saved step

`distillation/compare.py` evaluates the teacher and both saved students sequentially on one GPU. The first call
creates a reusable manifest of 16 full four-frame windows selected from an **explicit CSV**. It fixes clip/frame
indices, Gaussian noise, GT initial history, pose, audio codes, bad-audio masks and the full configured codebook
count. Hashes of the conditioning tensors detect changed data on later calls. Use `--prepare-only` to create or
validate the manifest on CPU. A CSV is not claimed to be held out unless its separation from both training
corpora has been checked.

The prepared fallback cohort is under `/mnt/paramfs/param/outputs/stabilizer-dmd-20261015/evaluation`:
`common-evaluation.csv`, `data/`, and `paired-windows.json`. It contains eight clips from Richmond 8 and eight
from Richmond 12, with 16 distinct person UUIDs, selected deterministically from clean source rows. This is
**training data**, not a verified held-out set. `cohort.json` preserves source CSV hashes, original rows and host
provenance; `materialization.json` records the 48 cropped file hashes (90.736 MiB). Each clip retains its first
64 video/pose latent frames and 750 Mimi time codes across all 32 codebooks. The model consumes its configured
16 codebooks; waveform decoding retains the complete codebook set. No full source videos were copied.

`selection.json` fixes one four-frame probe window per clip. Passing `--selection /path/to/selection.json` while
creating the manifest preserves this balanced selection; later evaluations reuse `paired-windows.json`.
`cpu-validation.json` confirms all 16 production sampler prefixes and numerical probe windows agree in
video, pose and audio indexing, including the last frame's audio bounds. This preparation performs no GPU
evaluation and does not substitute for checkpoint comparisons or paired video inspection.

```bash
.venv/bin/python -m distillation.compare \
  --config experiments/distillation/stabilizer-plan1.toml \
  --csv /path/to/common-evaluation.csv --root-dir /path/to/evaluation-data \
  --manifest /path/to/paired-windows.json \
  --dense /path/to/plan1/run/step-000350 \
  --speedy /path/to/plan2/run/step-000350 \
  --output /path/to/comparisons/step-000350.json --check-merged
```

Repeat at logical steps 550, 750, 950, 1150 and 1200 with the same manifest. Different logical steps are rejected.
The loader reconstructs LoRA from checkpoint metadata and strictly loads the **unmerged** `student.safetensors`,
including base weights and adapters, so evaluation uses the trained function. Older weights-only snapshots are
valid for evaluation even after their optimizer shards are pruned. Completion markers are checked before loading.

For an independent evaluation host, `python -m distillation.eval_bundle create --source CHECKPOINT
--destination BUNDLE --checkpoint-id UUID` publishes a verified student-only bundle. It checks the complete
source manifest and file generations, copies only the student and metadata, hashes every bundled byte, and
retains the original completion manifest as `source.checkpoint.complete.json`. The bundle's own marker is
`evaluation.complete.json`; it is explicitly not a resumable training checkpoint. Verify a transferred copy
with `python -m distillation.eval_bundle verify BUNDLE`. Both numerical comparison and video export accept
these bundles, preserving original checkpoint UUID and source lineage without copying optimizer shards.

JSON reports include per-window, per-frame and aggregate RMS differences, relative L2 and cosine against the
other student and the teacher. They also compare `(student - teacher)` changes: shared teacher content can make
raw output cosine look high even when learned changes disagree.

Every real stabilizer velocity prediction is recorded with its call, round, substep, slot, absolute frame and
actual time. At two steps per band there are **44 predictions per window**: 32 on generated frames and 12 on
GT seed frames. Invalid padded heads are excluded. Reports compare every prediction and group them by call,
frame, time and slot; each comparison includes cosine, L2 difference, RMS and magnitude ratio. These predictions
follow each model's own trajectory, so differences include both the model and the latent input it produced.

An independent common-input probe evaluates 12 velocities per window: four stabilizer slots at three positions
within their clipped training-time bands. GT latents, their sampled noise, GT history noise, times, pose and raw
audio are identical for all three models. This separates velocity-field disagreement from trajectory divergence.
The version-2 manifest fixes these probe settings. The companion safetensors file contains `latents/`,
`trajectory_velocities/` and `common_input_velocities/` tensors for each model.

Keep `--check-merged` in the paired evaluation plan to quantify merged-versus-unmerged LoRA rounding on the same
stabilized outputs and both sets of velocity predictions. This records the actual discrepancy introduced by
the optional deployment video's merged weights. Exact unmerged inference remains the numerical diagnostic's reference.

The completed logical-step-350 comparison (200 student updates in each run) illustrates why both checks matter.
On the 16 balanced training-partition windows, unmerged speedy versus dense has common-input velocity cosine
0.999563 and relative L2 difference 2.9746%; the corresponding teacher-relative changes have cosine 0.944148
and magnitude ratio 1.02872. Across all 704 rollout velocities the relative difference is 5.1257%, and across
generated latents it is 6.3982%. These are diagnostics, not a held-out quality verdict.

BF16 merging itself changes the speedy model by 1.9554% in common-input velocity relative L2 and reduces its
teacher-relative update magnitude ratio to dense from 1.02872 to 0.87934. A full export audit verified all 580
tensors, including all 204 merged weights, against the specified FP32 merge followed by base-dtype rounding.
This is a measured arithmetic effect rather than a checkpoint mismatch. Keep the original merged previews as
a deployment variant; primary video comparisons should retain the adapter branches. Results and partition
summaries are in `evaluation/preview-step350/` beneath the experiment output root.

### Paired rendered videos

`distillation/render_pairs.py` prepares full-clip inference with the existing `bimodal/sampler.py`. It deduplicates
the manifest's clips, fixes the same `manifest.seed + CSV row` GPU-noise seed in both arms, and fixes the teacher
checkpoint, resolver steps and resolver renoising seeds. These are full open-loop clips beginning from a GT sink,
so their noise geometry and seeds are separate from the four-frame numerical probes. Both arms share all video
inputs and settings except the student checkpoint and output directory.

```bash
.venv/bin/python -m distillation.render_pairs \
  --config experiments/distillation/stabilizer-plan1.toml \
  --csv /path/to/common-evaluation.csv --root-dir /path/to/evaluation-data \
  --manifest /path/to/paired-windows.json \
  --dense /path/to/plan1/run/step-000350 \
  --speedy /path/to/plan2/run/step-000350 \
  --max-latent-frames 64 --resolver-steps 2 \
  --output /path/to/videos/step-000350
```

Without `--run`, this writes the two YAMLs and `render-plan.json`, and prints the commands without launching
inference or exporting weights. Add `--run` to execute. The resolver is always the same frozen pose-v3 step-200
teacher; the explicit `--resolver-steps` controls its common schedule. Output panels are **GT | dense | speedy**,
for both resolved videos and stabilized-latent previews, with the same decoded Mimi audio.
CPU preparation checks input lengths, latent geometry, codebook count and Mimi indexing, hashes every selected
input file, and rejects a prefix that would omit any selected numerical window.

The default `--lora-mode unmerged` adds explicit LoRA metadata to the stabilizer's configuration. The inference
loader strictly restores every adapter factor in its saved dtype, reuses the training `LoRALinear` and fused-QKV
operators, and keeps FP32 audio normalization weights. The cached sampler preserves the adapter branches;
it does not merge their updates into BF16 base weights. A checkpoint containing adapters without matching metadata
is rejected. The frozen teacher resolver receives no adapter configuration.

`distillation/check_inference_lora.py` compares real checkpoints loaded independently through inference and
canonical `BimodalNet`, checks every parameter value/dtype and the audio encoder, and evaluates fixed common-input
probes. At step350, both dense and speedy loaders and their predictions under identical cached layouts matched
exactly. Across six probes (two clips, three times, 24 velocity heads), cached versus full-window canonical
predictions differed by 0% for dense and 0.9740% relative L2 for unmerged LoRA. This remaining layout-dependent
arithmetic difference is separate from merge rounding; the rendered videos are not claimed to be bit-exact
full-window canonical predictions. Both video arms use the same cached inference layout. With `--compile-wave`,
the corresponding compiled-versus-canonical differences were 1.2451% for dense and 1.1907% for unmerged LoRA;
CUDA graph replay matched compiled outputs exactly across changed probe inputs in both cases. The fresh dense
render matched all eight original dense latent tensors exactly (four clips, resolved and stabilizer-only),
confirming that adding optional adapter support did not change its completed preview.

Use `--lora-mode merged` with a separate output directory to retain a deployment comparison. This explicitly
calls `distillation/export.py` to merge `W + (alpha/rank) B A` in FP32 and round once to the stored base-weight dtype.
Export provenance records source and output SHA256s, checkpoint UUID and LoRA settings.
`compare --check-merged` measures the resulting numerical effect; adapter tensors are never silently ignored.
CPU verification is available through `.venv/bin/python -m distillation.test_compare`. No rendered-quality claim
follows from velocity cosine alone. **Paired video quality is the primary comparison** at matching logical
checkpoints; cosine, RMS, velocity-field probes and measured merge rounding are supporting diagnostics.

For the early four-clip preview, add `--rows 0 1 8 9`: this takes two clips from each source partition and
preserves original CSV row indices, so the four video seeds remain 1234, 1235, 1242 and 1243. With a dedicated
eight-GPU evaluation node, `--devices 0 1 2 3 4 5 6 7 --run` assigns one arm/clip shard per GPU and keeps each
worker's log. Completed arms receive `render.complete.json` with checkpoint identity, settings, input/source
hashes and output hashes. `--reuse-complete` verifies this receipt before reusing an early dense preview when
the matching speedy checkpoint arrives. The ordinary sampler now explicitly encodes each resolver's retained 750-code conditioning
table; it must not substitute the shorter 128-code waveform prefix of a trimmed 64-frame clip. Student and
teacher audio encoders remain separate, and teacher tokens are cached only within each clip/table start.

`--backend streamed` is an optional eager preview fallback implemented in `bimodal/streaming.py` and
`bimodal/streaming_sampler.py`. It constructs weights on CPU, streams one transformer/audio block at a time,
and runs stabilization, teacher resolution, and CPU FP32 media decoding as separate phases. It retains FP32
audio normalization weights and disables autocast's weight cache for reused block storage. CUDA graphs and
compilation are disabled. The six-GiB process budget reserves one GiB for the CUDA runtime and caps the
PyTorch allocator at five GiB; actual process residency must still be measured on the target GPU. CPU tiny-model
tests are bit-exact against ordinary eager cached inference, including repeated passes and distinct consecutive
weights; this is not yet a measured full-model GPU memory or speed claim. The dedicated evaluation node uses
the standard compiled sampler, so previews do not share training GPUs.
