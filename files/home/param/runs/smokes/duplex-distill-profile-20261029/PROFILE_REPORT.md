# Pose duplex distillation: measured profile and optimization audit

**Mean step wall time was 23.8362 seconds across steps 3–6.** Pose depth decoding and the subsequent wait for ranks with more pose frames form a substantial part of the step. The reported 2.7584-second speech-decoder collective interval is not a measurement of pure network transfer time.

## Scope and evidence

This was a six-step scratch profile of the full-depth model on all 48 GPUs across landsend-1,2,3,4,5,12. It started from the production student and teacher checkpoints, retained the balanced objective and real production data, and did not resume the stopped production run. Conversations retained their normal lengths up to 7,500 frames; shorter examples and examples without valid pose remained in the batch.

The raw file contains all **288 rank-step records**. Primary statistics below use only **steps 3–6: 192 records**. Steps 1–2 are excluded from those averages to avoid initialization effects; their mean wall times were 28.0263 and 22.0289 seconds. This is a short measured baseline, not a long-run throughput guarantee.

The launcher passed `--profile-steps 6 --no-wandb`. All 48 ranks completed all six steps, all six workers exited successfully, no checkpoint was written, and the completion audit at **2026-10-29 10:15:15 UTC** found every GPU process gone and GPU memory at zero. That timestamp describes profile completion, not any subsequent standalone benchmark.

- [Raw records](/home/param/pose-runs/duplex-distill-profile-20261029/profile-all-ranks.json)
- [Machine-readable timing summary](/home/param/pose-runs/duplex-distill-profile-20261029/timing-summary.json)
- [Completion verification](/home/param/pose-runs/duplex-distill-profile-20261029/verification/completion-audit.json)
- [Source/configuration verification](/home/param/pose-runs/duplex-distill-profile-20261029/verification/source-config-audit.json)

The saved `baseline-source/` snapshot is the source of record for this run. Its hashes matched all six workers. It predates the later metric-namespace cleanup and contains the older metric aliases; current repository metric names must not be used to reinterpret the saved baseline.

## How to read these timings

Intervals come from CUDA events on the current stream, with synchronization at the end of each step and no synchronization inserted at individual stage boundaries. They include stream waits, communication waits, and gaps while the CPU launches work. They are not kernel-only timings. CPU dispatch intervals overlap GPU intervals; parent and child spans also overlap. Do not add either category to the parent totals.

FSDP all-gathers and gradient reductions remain inside the forward/backward intervals. This profile does not isolate their communication cost from compute, and it is not an NCCL bandwidth trace. Stage placement can also shift observed waiting into a later dependent span.

## Top-level stages

All rows below are non-overlapping parent spans. Seconds are means over the same 192 rank-step records; the percentage uses the 23.8362-second mean wall time.

| Stage | Mean seconds | Share of wall | Min–max seconds |
| --- | ---: | ---: | ---: |
| Data-loader wait | 0.0983 | 0.41% | 0.0001–0.3886 |
| Gradient reset | 0.0043 | 0.02% | 0.0033–0.0071 |
| Frozen Voxtral encoding | 0.3027 | 1.27% | 0.0757–1.6257 |
| Input preparation + count reduction | 0.9265 | 3.89% | 0.0029–1.5358 |
| Audio teacher forward | 1.9313 | 8.10% | 1.6649–2.1655 |
| Pose teacher forward | 1.9581 | 8.21% | 1.6702–2.1716 |
| Student forward | 3.7948 | 15.92% | 3.7157–3.9171 |
| Speech depth decoding + backward | 0.6869 | 2.88% | 0.2003–1.0875 |
| Pose depth decoding + backward | 5.6052 | 23.52% | 0.0000–8.2256 |
| Decoder reduction, clipping, AdamW | 2.7969 | 11.73% | 0.0545–8.8820 |
| Student backbone/tower backward | 5.4243 | 22.76% | 4.9014–5.8113 |
| Metric reduction and processing | 0.0038 | 0.02% | 0.0034–0.0046 |
| Main gradient clipping | 0.1105 | 0.46% | 0.0416–0.3038 |
| Gradient checks | 0.0116 | 0.05% | 0.0088–0.0160 |
| Main AdamW update | 0.0303 | 0.13% | 0.0225–0.3535 |
| **Sum of parent intervals** | **23.6854** | **99.37%** | — |
| Unattributed wall residual | 0.1508 | 0.63% | — |
| **Measured wall total** | **23.8362** | **100.00%** | **23.0822–24.4548** |

The residual is wall time minus the sum of parent event intervals. It includes uninstrumented boundaries, such as logging/control handling, but is not itself a separately measured CPU or network stage. Mean wall time of the slowest rank in each included step was 23.8369 seconds.

## Nested costs

These intervals are already included in their parent rows above. Missing pose-decoder child spans contribute zero to the all-record averages.

| Nested interval | Mean seconds | Parent |
| --- | ---: | --- |
| `audio_teacher/trunk` | 0.9713 | Audio teacher forward |
| `audio_teacher/speech_tower` | 0.1675 | Audio teacher forward |
| `pose_teacher/trunk` | 1.0388 | Pose teacher forward |
| `pose_teacher/pose_tower` | 0.1662 | Pose teacher forward |
| `student/trunk` | 0.9933 | Student forward |
| `student/text_head` | 1.3764 | Student forward |
| `student/speech_tower` | 0.8607 | Student forward |
| `student/pose_tower` | 0.3099 | Student forward |
| `dec_s/teacher_forward` | 0.1348 | Speech depth |
| `dec_s/student_forward` | 0.1443 | Speech depth |
| `dec_s/backward` | 0.4054 | Speech depth |
| `dec_p/teacher_forward` | 1.0918 | Pose depth |
| `dec_p/student_forward` | 1.1273 | Pose depth |
| `dec_p/backward` | 3.3710 | Pose depth |
| `dec_p/kl` | 0.0131 | Pose depth |
| `dec_s/gradient_allreduce` | 2.7584 | Decoder optimizer |
| `dec_p/gradient_allreduce` | 0.0186 | Decoder optimizer |
| `dec_s/adamw` | 0.0067 | Decoder optimizer |
| `dec_p/adamw` | 0.0066 | Decoder optimizer |

The student text-head interval includes its forward-time computation of gradients for projected hidden states and output weights. Pose-decoder backward includes activation recomputation from the per-layer checkpointing needed for memory. The JSON includes every nested interval, CPU dispatch statistics, call counts, and percentiles.

## Pose workload imbalance and the following collective

The speech decoder is updated first, but only after **both** speech and pose depth work finish locally. Ranks with no pose frames reach that first collective much earlier than ranks decoding 7,500 pose frames. The observed pattern is:

| Pose frames in a rank-step | Records | Mean pose-depth seconds | Mean following speech collective seconds | Sum of these means |
| --- | ---: | ---: | ---: | ---: |
| 7,500 | 102 | 7.9605 | 0.2501 | 8.2106 |
| 1–7,499 | 62 | 4.2618 | 4.3228 | 8.5846 |
| 0 | 28 | 0.0000 | 8.4317 | 8.4317 |

The slowest pose-depth rank in each included step took approximately **8.18–8.23 seconds**. Across all ranks, pose depth averaged **5.6052 seconds** because shorter and missing-pose rows do less work. The next speech collective averaged **2.7584 seconds**, ranging from **0.0164 to 8.8436 seconds**. The following pose collective averaged only **0.0186 seconds**.

This strongly indicates arrival imbalance accounts for much of the first collective interval. Calling its 2.7584-second mean “network cost” would be incorrect. The minimum interval also does not independently measure pure transfer: the span contains the collective, stream waiting, and the world-size division. Balancing work may move time between rows without reducing the step; any proposed schedule change needs an end-to-end measurement.

## Optimizations retained and gaps to investigate

**Retained:** FA3 variable-length attention in trunk/towers; fused Triton RMSNorm, residual-plus-norm, SwiGLU, and Q/K preparation; embedding-bag code feedback; BF16 compute with FP32 parameter-gradient reduction; fused AdamW; backbone/tower activation checkpointing; the fused `attn32` decoder kernel; and one frozen Voxtral encoding shared by student and audio teacher.

**Not carried over:** the original production trainer enabled `compile_decoder(..., whole=True, dynamic=True)`. The distillation runtime replaces decoder forward and does not call that compiled path. It retains `attn32`, but decoder layers, heads, and KL execute eagerly, with student layer checkpointing. A KL-compatible compiled path must preserve these memory and gradient requirements; enabling the original CE wrapper alone is insufficient.

**Text head:** production used 1,024-position chunks and in-place `addmm` accumulation. The baseline uses 32-position chunks and computes `dw.add_(_mm_accum(...))`. Each full output-gradient temporary has 151,936 × 5,120 FP32 elements: **3.1116 GB**. A 7,499-position example therefore makes 235 chunk iterations, versus eight at 1,024. Larger chunks increase distribution-buffer memory, and full KL needs teacher as well as student distributions, so a chunk increase requires memory and numerical validation. The measured student text-head mean is **1.3764 seconds**, not an assumed saving. The separate benchmark below measures this candidate in isolation; its results are not part of the 48-rank timing averages.

**Unused teacher capacity:** the monoplex teacher still contains an unused speech depth decoder: **575,210,496 parameters, 2.3008 GB in FP32 per GPU**, verified from its checkpoint metadata. Its pose-only head call also enters an FSDP head unit containing the unused language output matrix and projector before returning the final normalized hidden state. These are memory/communication opportunities, not measured speedups; pruning or bypassing must preserve checkpoint loading and model outputs.

**Data path:** the exact production recipe and this baseline both disable the decoded waveform cache and use three persistent loader workers per rank. FFmpeg decodes the source FLAC before slicing/padding/gain, and pose files are checked on read. Original workers also lowered their CPU scheduling priority; this runtime does not. The measured loader wait is **0.0983 seconds** on average, so the profile does not establish decoding as the dominant bottleneck. Input preparation includes a distributed count reduction, so its **0.9265-second** interval should not be labeled pure host-to-device copying.

No speedup estimate is claimed for these changes. Stage microbenchmarks, gradient checks, and another full 48-rank profile are needed before adopting them.

## Separate text-head microbenchmark

A subsequent isolated run on one H100 on **landsend-5** used deterministic synthetic Gaussian inputs with the production text-head dimensions: **7,499 positions × 5,120 hidden width × 151,936 vocabulary**, BF16. Each case had one warmup and three timed iterations. Measurements cover the complete custom-autograd forward and backward; they exclude the other model modules, FSDP, data loading, optimizer, and correctness checks. [Raw benchmark results](/home/param/pose-runs/duplex-distill-profile-20261029/text-head-benchmark.json).

| Accumulation variant | Positions per chunk | Median seconds | Peak allocated GB | Peak reserved GB |
| --- | ---: | ---: | ---: | ---: |
| Baseline temporary + add | 32 | 1.5314 | 11.312 | 13.088 |
| Baseline temporary + add | 128 | 0.4759 | 11.313 | 12.354 |
| Baseline temporary + add | 256 | 0.2967 | 11.313 | 13.443 |
| Baseline temporary + add | 512 | 0.2257 | 11.313 | 15.615 |
| Direct in-place addmm | 32 | 1.2769 | 11.313 | 11.786 |
| Direct in-place addmm | 128 | 0.4080 | 11.313 | 12.589 |
| Direct in-place addmm | 256 | 0.2595 | 11.313 | 13.600 |
| Direct in-place addmm | 512 | 0.2006 | 11.313 | 15.928 |

The measured head time fell from **1.5314 seconds at baseline chunk 32** to **0.2257 seconds at baseline chunk 512**, and to **0.2006 seconds with direct addmm at chunk 512**. These are isolated measurements, not an end-to-end training speedup or proof that the variant fits the full training memory schedule. Peak allocated memory in the isolated cases was about 11.313 GB; allocator reservations increased with larger chunks.

All CPU checks against an independent FP64 autograd reference and all BF16 GPU tolerance checks passed. GPU losses matched the baseline value in this synthetic test; gradients were not bitwise identical across chunk sizes. The largest observed gradient relative-L2 difference was 0.0005401. Real checkpoint activations and full distributed training still need validation before changing the runtime.

The benchmark source is preserved separately under `text-head-benchmark-source/`. It does not modify the saved `baseline-source/` or establish that production training has resumed.

## Numerical comparison and limits

The first six profile steps were compared with the stopped baseline run. Maximum absolute differences from [the comparison artifact](/home/param/pose-runs/duplex-distill-profile-20261029/baseline-numerical-comparison.json) were:

| Metric | Maximum absolute difference |
| --- | ---: |
| `train/loss` | 2.0058467e-05 |
| `train/text_loss` | 3.5113216e-05 |
| `train/audio_loss` | 1.7574344e-05 |
| `train/pose_loss` | 5.62904525e-05 |
| `data_loss/loss` | 3.5383236e-05 |

The aggregate KL difference was **2.0058 × 10⁻⁵**. The profiling change adds instrumentation and a bounded scratch-run mode; it is not intended to change the training objective. Agreement of these scalar metrics does **not** prove bitwise identical weights, identical gradients, or parameter-by-parameter equivalence.

Across the included records, allocated-memory peaks ranged from **63.06 to 64.76 GB**. The profile includes optimizer updates but no evaluation or checkpoint writing. It does not characterize checkpoint I/O, long-run allocator behavior, or the rest of the dataset.
