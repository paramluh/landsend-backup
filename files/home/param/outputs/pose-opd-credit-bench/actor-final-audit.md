# Actor performance and Amu optimization audit

Measured configuration: 40 H100 80GB GPUs in 20 pairs, 16 simultaneous trajectories per actor pair, 320 rows per update, 1,500 physical frames per row, 25-frame conversation prefill. There are 23,600 useful generated physical frames per pair. Audio and pose share each physical frame; counting both as separate useful frames would artificially double throughput.

## Timings and comparison

| Boundary | Slowest-rank wall time | Useful generated frames/s/pair | Relative to audited Amu production |
|---|---:|---:|---:|
| Actor construction through publication, capture, prefill, generation, readback, teardown and pair broadcast | 128.4486 s | 183.7311 | 1.4210x |
| Data/frontend preparation plus that entire actor phase | 151.7962 s | 155.4716 | 1.2025x |
| Amu production reference, updates24–110 | median rollout63.3169 s | 129.2940 | 1.0000x |

The second comparison is closer to matching phase boundaries: Amu encodes its input audio inside its rollout phase. The first is useful to isolate this implementation's actor cost. The actor timer includes opt-in bounded policy diagnostics; it excludes optional trajectory-cache persistence (~0.89s). Checkpoint loading and initial Adam allocation were outside the timed update. The measured23.22s mean preparation is a first-use measurement, not a demonstrated steady-state encoder cost.

Amu's reference uses14 pairs and normally28 lanes/pair (20 in its final wave), variable conversations averaging about23.54s with median maximum56.24s, VAD/EOB stopping, and no pose decoder. Its129.294 figure is conversation audio seconds divided by rollout wall time, multiplied by12.5 and divided by14. Ours excludes the first two prefilled seconds and generates text/audio/pose for fixed120s rows. These are useful production-normalized measurements, not a controlled kernel A/B test. The separate Amu ~94ms/B35 cool-GPU harness and README2.2x headline are different measurements, not additional multipliers to apply here.

Rank0 setup breakdown: payload exchange before build1.027s; actor build2.938s; publication/cache release/barrier4.345s; engine allocation0.514s; graph warmup/capture4.709s; adapter features, prefill, generation and readback110.586s; final diagnostic/teardown/broadcast4.280s. Prefill was not independently timed. Peak observed device use across pairs was68.33GiB, including the colocated learner process. Publication-cache release freed about1.48GiB on each device on rank0.

## The two light decoders already run concurrently

`infer/engine/ppm/fast_ppm.py:1338` enqueues text on GPU1, copies the text/D33 conditioning to GPU0 using an event, enqueues the pose graph on GPU0, then enqueues the speech graph on GPU1 without a host or device synchronization between those graph launches. Backbone work follows the required input-copy dependencies. Measured first32 graph replays average40.368ms speech and40.327ms pose: those branch times overlap and must not be added to obtain an80.7ms serial cost. Backbone segments average12.965ms and14.509ms; text averages1.006ms. These early-context graph times exclude setup and do not substitute for the measured end-to-end timer.

Each light retains31 sequential codebook decisions across28 layers, because book j depends on the actual sampled earlier books. Parallelizing those31 decisions would change the sampling distribution. Teacher-forced training does evaluate all depth positions with causal attention. Audio and pose teacher/learner passes are scheduled separately on each learner rank to control memory and collective ordering; actor concurrency does not imply concurrent learner modalities.

## Static shapes and cache lifecycle

Selected actor shapes are NI16,1,500 conversation frames,32 depth-cache positions, and a bounded static prefix/sequence capacity. Learner/teacher rows are padded to1,636 positions with unscored causal tail; actual conditioning and prefill positions are preserved. Packed decoder and text chunks use fixed selected sizes with safe tail padding. The native actor captures text, partial/full speech, partial/full pose and both backbone segments after two warmups per graph. Device position buffers are updated while the same graphs and KVs are reused throughout generation.

The actor and graphs are released before learner replay and reconstructed next update. Static shapes do not make these allocations persist across update boundaries. Keeping a roughly49GiB/device actor and a roughly56GiB learner phase resident together would require a different offload/sharing design on80GB GPUs. Compiled Triton/FlashAttention kernel caches are reused normally; the selected learner is not globally `torch.compile`d. Variable actual prefixes still require row-specific prefill work. No per-frame CPU readback occurs in normal generation; diagnostics add bounded synchronization only when explicitly enabled.

## Optimization matrix against Amu's complete README

| Amu item | Current status | Concrete implementation or difference |
|---|---|---|
| R1 | Partly implemented; stop-specific parts inapplicable | No per-frame host readback, CUDA graphs, independent batched sampling. Fixed two-minute rollouts require no Silero VAD, speculative stop/redo, or VAD thread pool. Native voice-prefix deduplication/cache is not implemented; unique two-second conversation prefixes cannot be reused across rows. |
| R2 | Implemented with deliberate precision differences | Packed QKV/gate-up, fused norm/RoPE/KV/attention, SiLU multiplication, residual+RMS, 32-position KV, CUDA-graph replay. Same light_kernels source. FP32 residual/norm/RoPE and unrounded attention probabilities differ from Amu FastLight BF16 path. Selected M16 native kernel launch tuning retains kernel arithmetic; inverse CDF uses FP64 accumulation/uniforms over existing FP32 weights. |
| L1 | CPU parking implemented; CPU prefetch added after V7 | Frozen frontend CPU parking and shared frozen audio latents are present, with no bulk latent-cache write. V7 reads/encodes eight rows sequentially. CPU row prefetch was added after V7 and passed real-row parity/lifecycle tests; final40-rank validation is running. That change is not included in the reported timing. Warm encoder batching measured only44ms benefit and changed BF16 outputs. |
| L2 | Implemented | BF16 cast before all-gather, FP32 gain preservation, manifest validation, final consensus, and explicit release of unused publication allocator cache. |
| H1 | Not ported or verified | No explicit NUMPY_MADVISE_HUGEPAGE=0/NUMA memory policy/GPU-socket affinity/page-cache eviction watcher found in package. Different deployment; no single-update stalls do not prove long-run immunity. |
| N1 | Implemented | Depth decoders remain unsharded across chunk loops and are resharded in finally. Selected replay also defers decoder gradient synchronization across its chunk loop. |
| N2 | Implemented in normal production path | No per-chunk scalar finite checks or host code-range validation. Phase-level validation remains. Opt-in categorical diagnostic copies deliberately synchronize bounded samples and are not the normal training path. |
| A1 | Not applicable to current local-only saves | No HF upload in the new trainer, so no blocking upload exists. An asynchronous external HF uploader is not implemented. |
| S1 | Implemented independently | Rank-local weights/Adam/RNG and cursor/run identity, completeness marker and retention; selected checkpoint interval differs from Amu five-update cadence. |
| E1 | Not applicable | No Genghis evaluation exists in this entry point or resume path. |
| O1 | Partly implemented | Credit/KL and policy-delta metrics and optional benchmark trajectory cache exist. Recurring production six-rollout archival is not implemented; this is observability, not a speedup. |
| T1 | Not implemented; also off in Amu production | Teacher prefix-KV caching was a rounding prototype disabled in Amu production. Variable prefixes and packed replay require an independent parity/performance test. |
| T2 | Not implemented; also off in Amu production | Two student passes remain: detached costs then differentiable replay. Exact cached teacher logits eliminate repeat teacher passes. Amu fused student depth prototype was not shipped/enabled in the referenced production run. |
| Optional compact/bucket | Not applicable to selected fixed workload | All sixteen actor lanes generate the same two-minute span, so there are no finished-lane holes to compact. Amu compact/bucket mode was default-off because recapture costs erased its gain. |

Amu FastLight and this actor use the same `light_kernels.py` source (SHA2561874ff7c2f0da9cf23cb6b67fee48c145eb2808d069381794340044fd9ed9790), but this is not a byte-for-byte transplant of its numerical path. Our current source model/replay requires FP32 residual/norm/RoPE; the selected actor also avoids BF16 probability rounding in attention. Packed projections, fused attention/cache work, fused SiLU multiplication and fused residual normalization are present. The remaining31×28 sequential structure is the actual decoder, not an accidentally retained Python-level eager implementation. CUDA graphs remove the Python launch loop from each replay.

Additional measured changes beyond the referenced Amu path include the per-instance M16 native-GEMV launch configurations, FP64 inverse-CDF sampling, full-batch teacher preparation with cached CPU teacher logits, batched pose-CFG teacher temporal passes, deferred decoder gradient reduction, and selected removal of depth activation recomputation. Native tuned GEMV configurations matched the original native outputs bitwise in bounded randomized matrix tests. A proposed SiLU/down fusion was rejected because it was slower. Plain batched F.linear down projection was not selected because it changed outputs despite local speed gains.

## Numerical agreement and its consequence for credit

All10,880 requested full-categorical probes were captured:1,280 text,4,800 audio and4,800 pose. They cover selected early and final positions and media books0,14,31, rather than every trajectory decision.

| Actor-to-replay categorical KL | Mean | Median | p95 | Maximum |
|---|---:|---:|---:|---:|
| Text | 0.00021552 | 0.00000121 | 0.00138354 | 0.00237924 |
| Audio | 0.00021590 | 0.00014553 | 0.00055494 | 0.00409948 |
| Pose | 0.00115092 | 0.00051910 | 0.00416403 | 0.08244280 |

Sampled-logprob mean absolute differences over the first128 sampled frames of all320 rows were0.007906 text,0.013899 audio and0.034895 pose; respective maxima0.249741,0.906919 and0.724984. All values were finite. Pose book0/14/31 mean KL was0.0003075/0.0013098/0.0018354. The worst pose probe was final model frame1499,book31,rank7,row5 (`ALOE/s2handover-none/bd1407fc-e0ff-4d45-b7bd-87dd36780278/seg00/c00`), with KL0.0824428 and maximum logit difference1.53125.

A controlled final-frame probe supplied bitwise-equal temporal/D33 roots and actual sampled codes to cached incremental decoding and a teacher-forced depth replay. Across all31 depth positions its mean/max KL was0.00009932/0.002064 for audio and0.00044011/0.030930 for pose. This demonstrates that decoder arithmetic/batch geometry itself contributes. The full-pipeline difference can additionally include temporal-root rounding and different production replay chunk geometry. It is not justified to attribute all remaining error exclusively to D33. Audits found matching D33 shift, FP32 slot-zero addition, codebook offsets and actual sampled histories.

The sampler draws the full natural T=1 categorical policy: existing FP32 softmax weights, FP64 cumulative sum/uniforms, no top-k/top-p restriction, and all151,682 executable text tokens plus2,048 classes per media book. Replay uses those actual sampled IDs and their causal ancestors. However, actor and replay floating-point policies measurably differ. The stored actor logprobs are diagnostic; the score-function credit term differentiates replayed logprobs and has no importance correction. Therefore this is not strictly unbiased with respect to the actor's exact floating-point distribution. Structural causal correctness and a shared real-arithmetic model do not establish numerical identity.

A monitored mixed-precision pilot is empirically defensible; there is no demonstrated missing-causality bug in these probes. Strict numerical on-policy equivalence cannot be certified, and the pose tail outlier deserves explicit disclosure and monitoring. Small mean per-decision KL alone does not bound the error of an eight-frame, multi-codebook credit window. Amu also accepted geometry rounding (its own FastLight comparison meanKL~0.0002846), but that is context, not a validated tolerance for this pose model. If literal actor/replay identity is required, that claim must remain blocked until kernels are matched or a separately validated correction is introduced. Do not silently change the requested credit loss to add importance weighting.

## Remaining frontend opportunity

A separate measurement on the same eight1,500-frame inputs found CPU `data.load` work about4.00s; eight sequential GPU encode+CPU-copy calls0.568s warm (2.933s cold); a single eight-row encoder batch0.525s warm (12.667s cold). Batched preprocessing was bit-identical, but encoder latents differed with relative RMS0.004366 and maximum absolute difference0.1875;76.2% of BF16 values differed. Batched encoding raised allocated peak from2.102 to3.841GiB. Its measured44ms warm gain does not justify that numerical/memory change now. CPU row prefetch was added after this timing run and passed real-row parity/lifecycle tests; final40-rank resident-frontend validation is running as V9.

The earlier23s combined preparation also included cold data/decode and first-use initialization/compilation, CPU latent copies, and~1.047s frontend offload. It must not be presented as an unavoidable23s encoder loop. Subsequent restore cost and long-run preparation timing have not been measured here. Retaining the frontend during actor work would add roughly1.8GiB weights to the observed68.33GiB device peak; concurrent encoding would further consume compute/bandwidth and transient memory. It might fit, but no overlap test establishes safety or a net gain, and the warm encoder upside is small. CPU prefetch can overlap without introducing an untested GPU arithmetic path.

## Evidence

- Final40-rank summary: `/home/param/outputs/pose-opd-credit-bench/v7-logs/summary.json` and per-rank event archives in that directory.
- Audited Amu production reference: `/home/param/outputs/pose-opd-credit-bench/amu-rollout-baseline.json`.
- This machine-readable audit: `/home/param/outputs/pose-opd-credit-bench/actor-final-audit.json`.
- Amu complete switch table: `/home/param/opd-posttrain-speedups/README.md:40`; FastLight implementation: `code/src/finetuning/intelligence/speech_preservation/fast_rollout.py:191`.
- Actor graph scheduling/capture: `/home/param/pose-worktrees/pose-opd-credit/infer/engine/ppm/fast_ppm.py:1338` and`:1143`; package rollout implementation:`finetune/pose_opd_credit/rollout.py`; same-root probe:`policy_probe.py`.

This actor audit introduces no source changes. Separately, the root authorized one final V9 synthetic maximum-shape validation of resident frontend plus CPU prefetch; it is not part of these V7 actor timings. Timing includes one actual update with multiple replay alternatives; longer-run throughput, checkpoint overhead and repeated cold/warm behavior remain separate measurements.
