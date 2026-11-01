# Full-GT performance experiments — 2026-10-20

Status: E2 is selected for production after completing all benchmark work on 40 ranks, with 16.779 s median rounds and 8.927 GiB minimum sampled free device memory. Its process teardown required a scoped controller stop after the completed receipt; this is distinct from a training failure. A2 and E also completed; A, B, C, and D failed memory checks. E is slightly faster but leaves less than 1 GiB of sampled headroom. Production launched at 05:48:58 UTC; remote W&B metrics and initial critic-warmup progress were verified at 05:51:41 UTC.

## Workload and measurement

These trials run the actual trainer on **40 H100 80GB GPUs across five nodes**, using real rows from `aloe-studio-no-blackbands-high-quality-no-crossbleed.csv` (14,414 chunks, 458 people). Each outer round performs **four distinct critic optimizer updates and one student optimizer/EMA update**, with 160 global rows per objective per update and balanced objective coefficients of 0.5. The teacher checkpoint is `step10424/model_ema.safetensors` from the joint-nine-reference assets; exact paths and hashes are in each saved configuration and epoch plan.

The source data loader retains microbatch size 2. Execution regrouping changes local model batch sizes without changing source identities, the four optimizer boundaries, sample weights, or global loss denominators. Benchmark mode uses real gradient reduction, clipping, optimizer updates, and EMA, but skips evaluation, checkpoint saves, and W&B. Production logging/evaluation costs therefore are outside these timings.

Each trial's `plan.json` and completed receipt are authoritative for its round count and excluded warmup. A/A2/B request eight rounds with the first two excluded; C/D/E request six with the first two excluded; E2 requests four with the first excluded. A2 times below are **medians across six measured rounds**, E across four, and E2 across three, with each round's timing reduced by maximum across ranks. GPU memory peaks are maxima across ranks and measured rounds, not medians. Sampled device-memory figures come separately from one-second `nvidia-smi` telemetry and include memory outside the PyTorch allocator. CUDA phase timers are nested: critic/student optimizer times are already included in their enclosing phases and must not be added again.

The updated production request is **8,000 student updates after the existing 150 critic-only warmup rounds**, with checkpoints every **800 student updates**, plus manual/final saves. The full corpus epoch remains 168,284 groups; `max_student_steps=8000` caps its unchanged cursor and sets the student/critic LR schedule horizons to 8,000/32,600 optimizer updates (8,150 outer rounds total). Base LR remains `2e-6`, student/critic LR warmups remain 100/200 updates, and the final LR ratio remains 0.1. Retention remains eight completed raw/critic/EMA weight sets and **only the latest optimizer state**. Checkpoint directory numbers still count outer rounds; student update 800 therefore occurs at outer round 950. E's snapshot includes this requested limit/cadence; benchmark mode still runs only its explicitly requested six rounds. CPU horizon, resume, cadence, legacy behavior, and retention checks passed 32 tests.

## Execution matrix

Batch sizes below are local rows per GPU. `block` means activation checkpointing by transformer block; compilation remains shared whole-block `torch.compile`.

| Trial | Detached pool per objective | Critic stabilizer | Critic resolver | Student stabilizer | Student resolver | Reference checkpoint | Pool/student layout |
|---|---:|---|---|---|---|---|---|
| A | 16 | 4 / block | 4 / none | 2 / block | 4 / none | Inherit objective policy | Padded / padded |
| A2 | 16 | 4 / block | 4 / block | 2 / block | 4 / block | Inherit objective policy | Padded / padded |
| B — failed in round 2 | 16 | 4 / block | 4 / none | 2 / block | 4 / none | block | Padded / padded |
| C — failed in round 1 | 16 | 4 / none | 4 / none | 2 / block | 4 / block | block | Padded / padded |
| D — failed in round 2 | 16 | 2 / none | 4 / none | 2 / block | 4 / block | block | Padded / padded |
| E — completed | 16 | 4 / block | 4 / none | 2 / block | 4 / block | block | Trimmed / padded |
| E2 — selected; benchmark complete, teardown interrupted | 16 | 4 / block | 4 / block | 2 / block | 4 / block | block | Trimmed / padded |

A and A2 use `full_gt_trim=false`, a Dynamo cache limit of 128, accumulated limit 256, and `compile_fail_on_limit=true`. Exceeding a compilation limit fails the trial instead of silently falling back to eager execution.

## Validated results

**A failed** in the first critic resolver forward at batch 4 with checkpointing disabled. The preceding stabilizer forward/backward completed. Failures appeared across all five nodes, with about 77.2 GiB allocated by PyTorch and only about 60–95 MiB reserved but unused. This was activation capacity exhaustion rather than a large unused allocator reserve. A has no complete-round throughput result.

**A2 completed all eight rounds and all five node supervisors exited successfully.**

| A2 measurement | Result |
|---|---:|
| Outer-round wall time, median | **17.457 s** |
| Detached critic pools, median | 4.271 s |
| Four critic updates including their optimizer work, median | 6.872 s |
| Student update including optimizer/EMA, median | 6.477 s |
| Stabilizer portion of pooled generation, median | 3.638 s |
| Student stabilizer generation / backward, medians | 1.025 / 3.242 s |
| Peak allocated / reserved GPU memory | 63.030 / 68.467 GiB |
| Dynamo graphs per rank | 66 on every rank |
| Graph breaks | 0 on every rank |
| New graphs and compiler timer increments in measured rounds | 0 |

A2's first round took **228.685 s**, including graph capture, compilation/cache lookup, execution, and optimizer work. The first round recorded maximum-rank Dynamo compiler time of 146.929 s and lazy backward compiler time of 37.561 s; these instrumentation spans are not a separate throughput measurement. A2 reused the node-local cache populated during A, so its startup is not a clean-cache comparison. Steady rounds 3–8 had zero increments in both recorded compiler timers as well as zero new graphs.

`unique_graphs` counts Dynamo FX segments handed to the compiler backend per process. It does not count complete model architectures, AOT forward/backward executables, or generated GPU kernels. Different model weight values can share compiled functions; execution mode, tensor metadata, layouts, and batch shapes still specialize them.

**B failed in its second outer round.** Selective reference checkpointing allowed one complete round, reporting 76.007 GiB peak allocated memory and 62 graphs. That first round took 94.026 s including compilation and is not a steady-state result. In round 2, the student resolver's uncheckpointed batch-4 graph remained live while teacher reference prefill ran; teacher RoPE allocation then failed at about **76.84 GiB allocated**, with only 83.89 MiB reserved but unused in the recorded rank-5 failure. All five node supervisors exited unsuccessfully. Passing one round at this memory level was insufficient, and B supplies no completed steady-state benchmark.

**C failed in its first critic stabilizer forward**, despite checkpointed reference prefill. Its uncheckpointed batch-4 target pass reached about **77.29 GiB allocated** and failed a 118 MiB allocation. There was no completed round or steady-state result. This trial tested batch 4 before reducing the stabilizer microbatch.

**D reduced only the critic stabilizer microbatch to 2**, accumulating twice per objective and retaining its uncheckpointed target policy. One complete round reported 75.606 GiB peak allocated, 66 graphs, and 108.306 s including compilation. In the second round, after optimizer state had been initialized, critic stabilizer forward failed at about **76.82 GiB allocated**. D also has no steady-state result. Both C and D restored student resolver block checkpointing and kept reference-prefill checkpointing.

**E completed all six rounds on all 40 ranks**, with all five node supervisors exiting successfully. Its plan was recorded at 05:38:13 UTC and its completion receipt at 05:43:17 UTC. It restores critic stabilizer batch 4 with block checkpointing, retains critic resolver batch 4 without target checkpointing, keeps both student objectives checkpointed, and trims only the detached pool rollout. Reference prefill remains checkpointed.

| E measurement | Result |
|---|---:|
| Outer-round wall time, median | **16.607 s** |
| Detached critic pools, median | 3.684 s |
| Four critic updates including their optimizer work, median | 6.603 s |
| Student update including optimizer/EMA, median | 6.446 s |
| Stabilizer portion of pooled generation, median | 3.064 s |
| Peak allocated / reserved GPU memory | 76.485 / 76.592 GiB |
| Peak sampled total device memory used | 78.838 GiB |
| Minimum sampled device memory free | **0.810 GiB** |
| Dynamo graphs per rank | 82 on every rank |
| New graphs in measured rounds | 0 |

E's median round is about 4.9% shorter than A2's. Its observed memory margin is too narrow to infer long-run robustness from six rounds alone. The free-memory value above is measured by `nvidia-smi`; it is not obtained by subtracting device usage from the differently reported CUDA OOM capacity.

**E2 completed all four real-training rounds and is selected for production.** Its plan was recorded at 05:43:46 UTC and its all-40-rank completion receipt at **05:47:19.773 UTC**. It changes E's critic resolver target policy back to `block` and retains the trimmed detached pool, all other objective batch sizes, and reference checkpointing. It excludes the first round and reuses warm node-local compiler caches.

| E2 measurement | Result |
|---|---:|
| Outer-round wall time, median | **16.779 s** |
| Detached critic pools, median | 3.671 s |
| Four critic updates including their optimizer work, median | 6.787 s |
| Student update including optimizer/EMA, median | 6.435 s |
| Stabilizer portion of pooled generation, median | 3.054 s |
| Peak allocated / reserved GPU memory | 63.030 / 68.467 GiB |
| Peak sampled total device memory used | 70.721 GiB |
| Minimum sampled device memory free | **8.927 GiB** |
| Dynamo graphs per rank / graph breaks | 78 / 0 on every rank |
| New graphs and compiler timer increments in measured rounds | 0 |

E2 takes about **1.0% longer per round than E** and **3.9% less time than A2**, while recovering over 8 GiB of sampled device-memory headroom relative to E. Its sampled free memory is below the trial's initial 10 GiB target, but substantially above E's 0.810 GiB. These are short benchmark results rather than a long-run stability guarantee.

**Teardown caveat:** after the completed benchmark receipt, two workers retained CUDA allocations for over a minute while the other 38 released them. The controller issued a verified, trial-scoped stop at **05:48:24 UTC** to clear devices for production; node logs show the corresponding SIGTERM at 05:48:24.89, and the aggregated `stop.json` was written at 05:48:54. Thus the training benchmark completed, but process exit was interrupted and must not be reported as a clean all-supervisor exit. No data-phase exception or OOM preceded the receipt. The final per-round `gc.collect()`/`gc.freeze()` occurs before metric recording, the all-rank summary gather, and receipt writing, so the observed stall was not inside that fourth-round freeze. Remaining cleanup includes the completion broadcast, `gc.enable()`, process-group shutdown, and Python finalization. Installed PyTorch explicitly shuts down all registered groups in `destroy_process_group(None)`, including the teacher subgroups. No worker stack was captured at the stall, so the exact cleanup cause remains unproven; production code was not changed on that hypothesis.

## Implementation and next candidates

- **Corrections after the E2 benchmark:** the working tree removes the additional student scoring forward used only for diagnostics. Student velocity RMS now observes existing sampler outputs (`student_rollout_rms`, metric schema v2). It also caches the six fixed `.2` histories and sink once per stabilizer rollout, preserving their parameter-gradient connections; changing earlier-present states still recompute each step. These changes are not part of the E2 measurements or the immutable source of `training_8k`, and no GPU speedup is established for them yet. The previous implementation's repeated fixed-history work was avoidable overhead, not a requirement of GT conditioning.
- **Pool generation:** prepare the original microbatches first, preserving their audio-codebook augmentation granularity; concatenate all four critic updates into 16 rows per objective; generate with the fixed student under `no_grad`; slice endpoints and every retained context tensor back into the original updates. Random-number ordering changes with batching, while the sampling distributions and source identities are preserved.
- **Objective policies:** independently choose critic/student stabilizer/resolver microbatch sizes and checkpoint modes. Policy changes retain the shared compiled block callable. Student accumulation and four sequential critic updates retain their original normalization.
- **Selective reference recomputation:** optional `reference_checkpoint=block` checkpoints only the nine-frame reference prefill, restoring the objective policy before target execution. Cached reference K/V remains differentiable. CPU tests verify outputs, every parameter gradient, reference-input gradients, and restoration after exceptions. B passed critic resolver batch 4 without target checkpointing, but its student resolver exhausted memory during teacher scoring. C/D further establish that uncheckpointed critic stabilizer target passes at batch 4/2 exceed capacity; E retains block checkpointing for those paths.
- **Pool-only trimming:** optional `full_gt_pool_trim=true` with `full_gt_trim=false` trims future slots only during detached pool generation. Student generation and scoring stay unchanged. CPU tests with the real tiny BimodalNet verify output/context/RNG parity and single context preparation. E's stabilizer pool phase measured 3.064 s versus A2's 3.638 s; the full-trial comparison also includes changed checkpoint policies, so its overall speedup does not isolate trimming alone.

The earlier compile-only probe favored shared whole-block execution over parts compilation: trimmed block / padded block / trimmed parts had 76 / 40 / 45 graphs and warm synthetic-cycle times of 8.21 / 8.80 / 11.08 s. Those probes omitted optimizers, EMA, and cross-node gradient synchronization, so they guide candidate ordering rather than establish production speedups. Actual graph counts are taken from the production benchmark: A2 has 66 and E has 82; their differing checkpoint policies also affect that comparison.

## Evidence and reproducibility

All trial evidence is under `/mnt/paramfs/param/outputs/fullgt-performance-20261020-0505/`:

- A: `a_pool16_stab4cp_res4/`; first OOM stack is in `node-0/train.log`; `cluster.complete.json` records failed exits.
- A2: `a2_pool16_stab4cp_res4cp/`; completed results are in `runs/benchmark-20261020-052021-443390/benchmark.complete.json`, with per-round data in `metrics.jsonl` and actual topology/configuration in `runtime.json`, `config.json`, and `epoch-plan.json` alongside it.
- B: `b_pool16_stab4cp_res4_refcp/`; the completed first round is in `runs/benchmark-20261020-052747-550505/metrics.jsonl`; `node-0/train.log` identifies student resolver teacher-prefill OOM, and `cluster.complete.json` records all failed exits.
- C: `c_pool16_stab4none_res4none_refcp/`; `node-0/train.log` identifies first-round critic stabilizer OOM; `cluster.complete.json` records all failed exits.
- D: `d_pool16_stab2none_res4none_refcp/`; first-round observations are in `runs/benchmark-20261020-053428-769177/metrics.jsonl`; `node-0/train.log` identifies second-round critic stabilizer OOM, with failed exits in `cluster.complete.json`.
- E: `e_pooltrim16_stab4cp_res4none_refcp/`; completed results are in `runs/benchmark-20261020-053922-979620/benchmark.complete.json`, with per-round data in `metrics.jsonl`; `cluster.complete.json` records all-node success. The aggregate `summary.json` records the 78.838 GiB sampled device-memory peak.
- E2: `e2_pooltrim16_stab4cp_res4cp/`; completed results are in `runs/benchmark-20261020-054454-726866/benchmark.complete.json`, with four rounds in `metrics.jsonl` and all 40 rank summaries. `node-1/train.log`, `node-4/train.log`, `stop.json`, and `watchdog.json` record the later controller-induced teardown interruption; those exit statuses do not invalidate the earlier benchmark receipt.
- Every trial saves immutable source in `code/`, with per-file SHA-256 hashes and base commit in `code/source-manifest.json`. Base commit for A/A2 is `b8a8c2b030d155fd5c3ffd42dc5edcfeb6fe3864`; the source manifest also captures uncommitted implementation changes.
- `plan.json` records overrides; `node-{0..4}/spec.json` records the exact command and environment; `gpu.csv` records one-second utilization, memory and power samples; `train.log`, `process.json`, and `exit.json` record progress and process outcomes. `cluster.complete.json` records all-node success/failure.
- Environment: PyTorch 2.11.0+cu130; `OMP_NUM_THREADS=4`, `TORCHINDUCTOR_COMPILE_THREADS=8`, `PYTORCH_ALLOC_CONF=expandable_segments:True`, NCCL/Gloo interface `eth0`. Local caches are `/tmp/param-fullgt-training-20261020/{inductor,triton}` and are reused between trials on each node.
- Earlier graph-count evidence: `/mnt/paramfs/param/outputs/full-gt-compile-probes-20261020-025915/compile-summary.json` and its per-variant receipts.

## Final selection and production receipt

Production was launched at **05:48:58 UTC** under `/mnt/paramfs/param/outputs/fullgt-performance-20261020-0505/training_8k`, using E2's selected execution settings and the requested 8,000-student-update horizon, 150 critic warmup rounds, and 800-student-update save cadence. **Only the detached critic-training rollout pool is unpadded** (`full_gt_pool_trim=true`); the gradient-carrying student rollout remains padded (`full_gt_trim=false`), as does scoring. All critic/student target passes use block checkpointing, and reference prefill uses block checkpointing.

The live run is `training_8k/runs/run-20261020-055010-607506/` under that root.
[W&B run](https://wandb.ai/canopy-labs/param-distillation/runs/248d0032ed70)
was verified through the remote API at **05:51:41 UTC**: state `running`, with
training steps 1 and 2 and finite critic FM losses visible in remote history.
The local counters then showed two critic-warmup rounds and eight critic
updates; student updates remain zero during the preserved headstart. The
second warmup round took 10.349 seconds with 42 graphs and no new compilation.
All five nodes were observed running eight GPU workers, with no exit receipts.
Remote evidence is saved in `training_8k/wandb-remote-verification.json`.

At the selected 16.779-second full-round median, 8,000 student rounds plus the
critic headstart project to about **38 hours**, excluding evaluation and save
overhead. First scheduled evaluation/save is at student step 800. Full GPU
evaluation and parallel checkpoint I/O will first be exercised then; benchmark
mode excluded both. CPU regression and horizon checks passed (121 checks across
the recorded invocations); details and the corrected test-module invocation
are in `validation.json`. Installed evaluation dependencies and all-node import
checks are recorded in `env-preflight.json`. Cloudflare publication still awaits
the user's VM address; local galleries and W&B publication are implemented.
