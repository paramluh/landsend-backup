# Two-minute, batch-320 validation (2026-10-31)

The selected recipe uses the first 1500 physical frames of every admitted row,
25 unscored prefill frames, global batch 320, 16 concurrent actors per GPU pair,
and eight learner rows per GPU. Rows shorter than 1500 frames are explicitly
excluded; none of the currently eligible 13,566 orientations are shorter.

## Completed on-policy benchmark

Run `pose-opd-bench-b320-static-v7` completed one unsaved Adam update on all
40 H100 GPUs (landsend-2/3/4/5/12), with all five supervisors exiting zero.
Adam moments were allocated before rollout. One set of 320 actual student
trajectories was reused for four replay trials without intervening weight
updates; only the last trial supplied the actual optimizer step.

| Student rows/GPU | Depth checkpointing | Decoder gradient reduction | Depth/text chunks | Slowest rank replay |
|---|---|---|---|---|
| 8 | enabled | every chunk | 128 / 32 | 67.10 s |
| 8 | disabled | end of replay | 128 / 32 | 38.63 s |
| 4, accumulated twice | disabled | end of replay | 256 / 128 | 38.40 s |
| **8, selected** | **disabled** | **end of replay** | **256 / 128** | **35.85 s** |

Backbone and tower activation checkpointing remain enabled. Trial order was
fixed, so first-use compilation/allocation can disadvantage the first trial;
the 67-to-36-second comparison is not an isolated checkpointing experiment.
The matched final two trials favor physical batch eight by about 6.7%.

| Phase | Slowest rank |
|---|---|
| Row loading, frozen audio features, copies/offload | 23.38 s |
| Actor publication, capture, prefill, generation, readback and release | 128.45 s |
| Shared audio teacher batch 8 and pose CFG batch 16 | 18.19 s |
| Selected student replay/backward | 35.85 s |
| Gradient clipping and Adam | 2.26 s |

The maximum per-rank sum of those selected phases is **207.38 seconds (3m27s)**.
This is an estimated selected-configuration iteration, not a separately timed
production iteration: the actual diagnostic job took 358.70 seconds because it
ran four replay trials plus diagnostics. Model/process startup is excluded.
The shared teacher cache in this comparison was built with depth chunks 128;
the maximum-shape follow-up tests the selected teacher chunk size 256 as well.

Maximum sampled total device usage was **70,467 MiB (68.82 GiB)**, including
resident Adam state, leaving about 10.8 GiB of physical headroom. Phase-specific
NVML peaks were 38.23 GiB for teacher scoring and 68.33 GiB for the actor.
The selected replay's PyTorch allocation peak was 56.54
GiB and reservation peak 66.13 GiB; these are different memory measurements.
Final mean weighted KL was 0.03738318 and global gradient norm 0.9504649, both
finite. The selected gradient L2 summary differs by 0.0437% from trial zero;
summary checksums do not establish elementwise gradient equality. Dedicated
two-rank FSDP tests separately cover deferred reduction and accumulation.

The selected replay's rank-zero CUDA stage measurements were approximately:
17.93 seconds for both media depth forward/backwards, 5.95 seconds for final
temporal backward, 6.19 seconds for the two student temporal forwards, and
5.41 seconds for detached media scoring. Dense credit construction itself took
2.7 milliseconds. The remaining dominant cost is sequential rollout generation,
followed by actual transformer forward/backward work; this does not establish
that no further optimization is possible.

## Maximum-shape stress test

`pose-opd-bench-b320-static-adversarial-v8` completed on all 40 ranks with all
five supervisors exiting zero. Every slot used an actual maximum-length
conditioning prefix: **134 + 1500 + 2 = 1636 occupied positions**, so all 13,088
student positions were real timeline positions rather than appended padding.
The selected batch-eight learner, teacher batch eight/CFG sixteen, depth chunks
256 (512 combined CFG branches), disabled depth checkpointing, deferred gradient
reduction and preallocated Adam state all fit. Teacher scoring took at most
17.17 seconds, student replay 35.95 seconds, and clipping/Adam 2.15 seconds.
Total sampled device usage peaked at **69,606 MiB**, with finite gradients.

This test retained real conditioning/audio/prefill and cycled valid prefill B
tokens to fill the generated region. It deliberately skipped actors and policy
comparisons. It establishes capacity and executable backward/optimizer behavior;
its loss/gradient magnitude is not a rollout quality or learning result.

A final follow-up, `pose-opd-bench-b320-resident-prefetch-v9`, repeated the
maximum-shape update with the CPU prefetch worker and the frozen audio encoder
kept resident. All 40 ranks and five supervisors completed successfully. Peak
device usage was **71,488 MiB (69.81 GiB)**; teacher/replay maxima were 17.67/36.88
seconds. The first prefetched batch waited at most **72 microseconds** after
model setup. Preparation took 3.66 seconds. This synthetic test validates the
final resident-encoder learner configuration; it does not remeasure on-policy
actor throughput or long-run preparation latency.

## Audio preparation and overlap

The first-update 23.38-second preparation measurement combines CPU row loading,
waveform decoding, encoder execution, CPU copies, offload and distributed
waiting. It must not be described as 23 seconds of steady-state encoding.
A standalone test on the exact eight v7 rank-zero waveforms measured about
4.00 seconds for CPU loading and **0.568 seconds for eight warm sequential
encoder calls plus CPU latent copies**. The cold encoder pass took 2.93 seconds.

The native batched encoder was tested with those identical inputs. Its warm
time was 0.525 seconds, only 44 ms faster, while cold time rose to 12.67 seconds
and allocated memory from 2.10 to 3.84 GiB. Resampling and log-mel preprocessing
were bit-identical; encoder output relative RMS difference was 0.004366 and
maximum absolute difference 0.1875. This batching change was rejected. The
existing encoder remains the measured, numerically consistent path.

The selected `prefetch_rows=true` now loads the next effective batch in a
persistent spawned CPU-only worker while the current update runs. The first
batch is submitted before model loading. The queue holds exactly one future
batch, preserves the saved plan/resume cursor and dummy-row mapping, and
propagates load failures through the existing all-rank error agreement. Process
isolation is required because the archived loader temporarily changes Python
imports and environment variables. Exact equality of eight real loaded rows,
waveforms and conditioning was verified against the synchronous loader.
GPU feature encoding remains on the training rank; it is not claimed to run
concurrently with the actor. The benchmark above predates this CPU overlap.

## Rollout throughput and Amu comparison

The audio and pose light decoders overlap across the two actor GPUs. Across
20 pairs, their measured stage durations average 40.37 and 40.33 ms, respectively;
they do not contribute 80.7 ms in series. The two backbone portions average
12.97 and 14.51 ms, and text scoring averages 1.01 ms.

Each pair generates 16 × 1475 = 23,600 physical frames after prefill. Throughput
is **183.73 useful generated frames/s/pair** including the complete actor phase,
or **155.47 frames/s/pair** including the preceding data/audio preparation.
The audited Amu production reference is **129.29 conversation frames/s/pair**.
This is comparable or better throughput, but not a controlled same-workload
speedup: Amu's audio-only run used 28 lanes per pair, short exchanges and early
termination, and its frame accounting includes question/prefill context. This
run includes a pose tower/light and uninterrupted two-minute trajectories.

The actor uses native compiled Triton kernels, CUDA graph capture, cached
backbone/depth attention, concurrent audio/pose branches, and 16 active lanes.
The depth caches reserve 32 positions. The selected M16 linear specialization
changes native kernel launch configurations while retaining the kernel body;
tested tuning candidates were compared with the native outputs. Inverse-CDF
sampling uses FP64 cumulative sums and uniforms while retaining the full
temperature-one categorical law. No top-k, truncation or CFG is applied to the
student. Native fast-light optimizations and additional source-audit details
are recorded with the accompanying benchmark artifacts.

## Static shapes and numerical limits

Every learner/teacher segment has **1636 positions**: the maximum native
reference prefix (134), conversation (1500), and media-delay flush (2).
Shorter native reference prefixes receive an unscored causal tail; their real
token positions are unchanged. Student/audio-teacher batch eight has 13,088
positions, and the packed conditional/unconditional pose teacher has 26,176.
Actor batch size and maximum cache capacity are fixed across all GPU pairs.
Padding does not replace missing conversation data: short conversations are
excluded. Static shapes stabilize kernel/cache specialization; the complete
FSDP training step is not claimed to be one compiled CUDA graph.

Cached actor and full-sequence replay use different BF16 kernel geometries and
are **not bit-identical**. All 10,880 requested full-categorical probes were
captured. Mean actor-to-replay categorical KL was 0.0002155 for text,
0.0002159 for audio and **0.0011509 for pose**; the worst sampled pose probe was
**0.08244**. A separate decoder comparison with bitwise-identical root inputs
still measured mean audio/pose KL 0.0000993/0.0004401. That probe uses depth
batch 16 rather than production chunk 256, so it cannot attribute all remaining
discrepancy exclusively to temporal roots. These are measured numerical
differences, not evidence of exact on-policy arithmetic or gradient parity.
The trainer records sampled log-probability discrepancies on normal updates.

A proposed `torch.compile` RMSNorm optimization was rejected despite its local
speedup because BF16 forward and gradient rounding changed. The measured
native linear configurations were retained instead. No precision reduction
or sampled-KL approximation was introduced to accelerate the objective.

## Reproducibility

Local evidence root: `/home/param/outputs/pose-opd-credit-bench`.

- `v7-logs/summary.json`: all-rank phase/trial results, stage timings, policy
  probes, memory, source/plan identity and hashes of archived evidence.
- `actor-v7-summary.json`: actor stage/concurrency and identical-root probes.
- `actor-final-audit.md`: source-by-source Amu optimization matrix, including
  inapplicable stop/VAD optimizations and unported host-memory tuning.
- `v8-logs/summary.json`: full occupied maximum-shape stress test.
- `frontend-batch/result.json`: exact-input cold/warm encoder comparison.
- `amu-rollout-baseline.json`: production reference and accounting definitions.
- `gemv-m16-tune.json`: native linear launch-configuration measurements.
- `first120s-dataset-accounting.json`: selection and epoch accounting.
- `cpu-tests-launch-final.log`: 171 passing tests including launch configuration,
  three-epoch LR/checkpoint policy, isolated CPU prefetch and sample exports.
- `final-three-epoch-plan-schedule.json`: all 129 sample selections, epoch seeds,
  checkpoint cadence and complete Amu learning-rate curve.
- `v9-logs/summary.json`: resident-encoder plus CPU-prefetch maximum-shape run.

Production launch helpers are in `/home/param/outputs/pose-opd-credit-launch`.
Preparation stages a clean exact commit on the five selected hosts and never
starts training. The separate start command validates source identity and idle
GPUs. Its normal training command has no benchmark limit, optimizer warmup or
`--no-save`; production checkpoints and normal Adam initialization are enabled.
