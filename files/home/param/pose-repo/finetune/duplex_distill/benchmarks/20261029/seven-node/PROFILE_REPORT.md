# Seven-node microbatch comparison — 2026-10-29

**Microbatch two is selected.** On 56 H100s, it processed 30.98% more real examples per second than microbatch one. Its measured steady update took 25.1921 seconds, versus 16.4987 seconds for batch one, while processing twice as many rows. The two-epoch projection is **1.99 hours versus 2.60 hours**, excluding loading and checkpoint overhead. Maximum allocated memory increased from 62.1604 GB to 68.8627 GB.

## What was measured

Both runs used landsend-0,1,2,3,4,5,12, eight GPUs each, full production model depth, fresh production student/teacher weights, and the actual training forward/backward/optimizer paths. Each completed six scratch updates on every rank, with no W&B, evaluations, or checkpoints. Both use balanced pure forward KL at temperature one, shared 10% speech/pose depth sampling, CB0 on every valid frame, and text chunk 512 with direct-addmm accumulation. There is no added conversation crop: each row retains its production limit of 7,500 conversation frames, plus prefix/delay. Naturally short rows remain short.

Microbatch two prepares two distinct conversations per rank, encodes each waveform independently, then packs them with isolated attention segments, reset positions, independent prefixes, and masks. Its global valid-position denominators span both rows across all 56 ranks. This is a real batch of 112 examples per update, without gradient accumulation. Both complete plans cover 31,796 real oriented examples and 12 zero-weight padding slots across two epochs. The plan digest is `05ce3e0b2bbe519f` in both runs.

The primary window is **steps 3–6**, excluding two initial updates: 224 rank-step records per run, comprising 224 real rows for batch one and 448 for batch two. For each step, take the maximum wall time across all ranks; the headline time averages those four maxima. Throughput divides global row/frame totals by the sum of those maximum times. All raw records, including initialization-sensitive steps 1–2, are retained.

## Throughput and memory

| Measurement | Microbatch 1 | Microbatch 2 |
| --- | ---: | ---: |
| Global examples per update | 56 | 112 |
| Mean maximum-rank update wall time | 16.4987 s | 25.1921 s |
| Global real examples per second | 3.3942 | 4.4458 |
| Conversation frames per second | 20,618.7 | 26,267.4 |
| Timeline positions per second, including prefixes/delays | 20,679.5 | 26,343.6 |
| Selected speech depth frames per second | 2,063.5 | 2,623.9 |
| Selected pose depth frames per second | 1,794.4 | 2,276.8 |
| Maximum allocated memory, all six steps | 62.1604 GB | 68.8627 GB |
| Two-epoch optimizer updates | 568 | 284 |
| Warmup updates | 19 | 9 |
| Projected two-epoch update time | 2.6031 h | 1.9874 h |

Batch two improves conversation-frame throughput by **27.40%** and projects **23.65% less update time** for complete dataset coverage. The projections are `568 × 16.49867755` and `284 × 25.19211428` seconds. They omit model loading, two-hour/final checkpoint costs, and variation across the remaining dataset; four steady steps are a bounded sample, not a full-epoch measurement. Memory is CUDA allocated high-water mark, not reserved memory or total device usage.

| Step | Batch-one maximum rank wall | Batch-two maximum rank wall |
| --- | ---: | ---: |
| 1, excluded | 48.9044 s | 31.2126 s |
| 2, excluded | 17.3481 s | 22.8728 s |
| 3 | 16.7833 s | 25.4881 s |
| 4 | 17.5588 s | 25.7701 s |
| 5 | 16.4246 s | 24.5565 s |
| 6 | 15.2280 s | 24.9538 s |

## Where time goes

These are **mean CUDA-event parent intervals over all 224 steady rank-steps**, not maxima. They include current-stream idle time caused by CPU dispatch and communication/arrival waits. Nested intervals overlap their parents. FSDP collectives are folded into forward/backward stages; this is not an isolated network trace.

| Parent stage | Batch 1, seconds | Batch 2, seconds |
| --- | ---: | ---: |
| Data-loader wait | 0.0941 | 0.1686 |
| Zero gradients | 0.0042 | 0.0042 |
| Frozen audio encoding | 0.3047 | 0.4791 |
| Prepare inputs and global counts | 1.6353 | 0.9173 |
| Frozen audio-teacher forward | 1.8192 | 3.4267 |
| Frozen pose-teacher forward | 2.0589 | 3.3522 |
| Student backbone + towers + text/CB0 KL | 2.5971 | 4.0934 |
| Speech depth teacher/student/KL/backward | 0.6854 | 1.3057 |
| Pose depth teacher/student/KL/backward | 0.5781 | 1.1170 |
| Decoder gradient reductions, clipping, updates | 0.6947 | 1.1039 |
| Student backbone/tower backward | 5.7534 | 9.0522 |
| Metrics | 0.0024 | 0.0025 |
| Main gradient clipping | 0.0436 | 0.0504 |
| Gradient checks | 0.0109 | 0.0115 |
| Main optimizer | 0.0263 | 0.0254 |
| Sum of parent means | 16.3083 | 25.1099 |
| Mean rank wall minus parent sum | 0.1899 | 0.0818 |

The student forward row includes input embeddings/adapters, trunk, speech/pose towers, pose link, text-gradient precomputation, and both CB0 KL losses. The text-head nested interval averages 0.2392/0.5136 seconds for batches one/two. Speech depth backward averages 0.4062/0.7777 seconds; pose depth backward averages 0.3484/0.6719, including zero on rank-steps without selected pose.

Input preparation is dominated by the count-collective interval: 1.6310 seconds for batch one, 0.9065 seconds for batch two (two calls summed). Actual CPU count construction averages 0.161/0.283 milliseconds. The three role-array transfer intervals total 3.018/6.021 milliseconds; batch-two packing totals another 2.313 milliseconds across roles. The first array transfer’s CPU dispatch can block behind the preceding collective, so its 1.6321/0.9088-second CPU interval is not a measurement of copying arrays alone. The lower batch-two count interval does not establish faster network bandwidth.

The first decoder gradient collective averages 0.6575/1.0661 seconds, but minima are approximately 17 milliseconds in both runs. These intervals include ranks waiting for others to finish different amounts of depth work; they cannot be interpreted as pure network costs. The following pose-decoder collective averages 17.53/18.01 milliseconds. No speedup is attributed to an isolated kernel or communication subsystem without corresponding measurements.

## Coverage and validation

Across all six steps, batch one processed 336 real rows, with 285 pose-present and 51 missing-pose rows. Batch two processed 672, with 575 pose-present and 97 missing-pose rows. In the steady window the counts are 191/33 and 383/65. Every pose-present row has equal speech/pose sampled-frame counts; missing pose has zero selected pose frames. Exact index alignment is enforced by the data adapter and CPU tests, not inferred from count equality alone.

Both completion audits pass all 27 checks: all 56 ranks completed six updates; all seven launchers exited zero; losses remained finite; teacher gradients were absent; monoplex A conditioning stayed native-empty; full architecture, shared sampling, objective weights, plan, source hashes, and configuration matched. GPUs were idle at each run’s completion audit; subsequent production training is a separate launch. The batch-two packer also has a supplemental all-node hash capture explicitly timestamped after launch. Captured forward/training source files match between runs; the only difference among shared captured files is a checkpoint-robustness change, whose save path neither profile executes.

The two steady windows consume different portions of the same deterministic per-rank row stream. Mean conversation length is approximately 6,074.7 versus 5,908.3 frames. The batch changes optimizer grouping, warmup, and update count; this benchmark does not establish identical gradients, numerical trajectories, or final convergence. CPU tests separately verify packed attention isolation and globally normalized KL gradients, including unequal lengths, missing pose, and dummies. Neither six-step profile exercises checkpoint writes or complete-epoch boundary padding.

## Reproduction and evidence

Run `python analyze.py` from this directory or pass its path from the repository root. It rebuilds [timing-summary.json](timing-summary.json), including all stage statistics, per-step totals, projections, and SHA256 hashes of retained inputs. Raw data are unchanged:

- Batch one: [profile](mbs1/profile-all-ranks.json), [configuration](mbs1/config.json), [plan](mbs1/plan.json), [metrics](mbs1/metrics.jsonl), [completion audit](mbs1/verification/completion-audit.json), [source audit](mbs1/verification/source-config-audit.json), and captured source under `mbs1/verification/launch-source/`.
- Batch two: [profile](mbs2/profile-all-ranks.json), [configuration](mbs2/config.json), [plan](mbs2/plan.json), [metrics](mbs2/metrics.jsonl), [completion audit](mbs2/verification/completion-audit.json), [source audit](mbs2/verification/source-config-audit.json), and captured source under `mbs2/verification/launch-source/`.

Original run directories are `/home/param/pose-runs/duplex-distill-7n-mbs1-profile-20261029` and `/home/param/pose-runs/duplex-distill-7n-mbs2-profile-20261029`. No scratch-profile checkpoint is available for resuming production.
