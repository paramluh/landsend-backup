# Production data coverage and packing semantics

Audited **October 29, 2026, 03:44 PDT** on landsend-12 using CPU metadata only. The pose index was joined to all four `rows.parquet` files and the exact saved training plan. File hashes, source references, cross-tabulations, and timestamps are in [data-coverage.json](/home/param/pose-repo/finetune/duplex_distill/benchmarks/20261029/data-coverage.json). Pose payloads were not rescanned; no training or data pipeline changed.

## Actual examples

Each distinct conversation contributes its first chunk in **both speaker orientations**. One oriented example runs on each rank per step. Short examples are **not concatenated or padded to 7,500 frames**. The complete model timeline is `T = P + n + 2`, where `P` is that example's conditioning prefix and `n` is its actual conversation length. The 7,500-frame limit applies to conversation frames, not the complete timeline.

| Split | Distinct conversations | Shorter than 7,500 | Full 7,500 | Oriented examples |
| --- | ---: | ---: | ---: | ---: |
| Train, one epoch | 7,949 | 3,523 (44.3200%) | 4,426 | 15,898 |
| Eval | 158 | 61 (38.6076%) | 97 | 316 |

Training conversation lengths range from 1,502 to 7,500, with mean 6,017.014 frames. Each orientation has the same length distribution: training contains 7,046 short and 8,852 full oriented examples per epoch.

## Pose coverage

| Split / B speaker | Oriented examples | Valid B pose | Missing or rejected B pose |
| --- | ---: | ---: | ---: |
| Train, assistant (`swap=0`) | 7,949 | 6,816 | 1,133 |
| Train, user (`swap=1`) | 7,949 | 6,762 | 1,187 |
| **Train, both** | **15,898** | **13,578** | **2,320 (14.5930%)** |
| Eval, assistant (`swap=0`) | 158 | 138 | 20 |
| Eval, user (`swap=1`) | 158 | 141 | 17 |
| **Eval, both** | **316** | **279** | **37 (11.7089%)** |

Among training conversations, 6,039 have both pose tracks, 1,500 have one, and **410 have neither**. Eval has 123 with both, 33 with one, and **2 with neither**: **412 conversations with neither across both splits**. Missing-pose examples remain useful for text and speech training.

Pose validity is **all-or-nothing for each oriented example**, not a partial frame mask. A valid track supplies all `n` conversation frames. Pose codebook-0 and higher-codebook targets occupy positions **`P+2` through `P+n+1`, inclusive**. This excludes the entire conditioning prefix and the first two conversation positions, and includes the delayed tail. Every accepted code is checked to lie in `[0, 2048)`; source track length must cover `n`. Rejected tracks produce `pdepth=-1`, `pcb0=-100`, and an empty `plight` over the entire example: no pose input or pose loss.

For training, pose higher-codebook coverage is **100% of those valid frames**. Speech codebook-0 also uses every conversation frame, while speech higher codebooks use the existing deterministic approximately **10% subset**. Text has `n−1` scored positions. These are different counts and must not be described as equal frame sampling. Eval subsequently caps each depth stream to 2,048 selected frames per row.

One training epoch contains **95,658,484** oriented conversation frames and **82,678,649** valid B-pose frames (86.4311% of conversation frames). The current two-epoch plan exposes **165,357,298 pose frames**, visits all **31,796 real oriented examples**, and adds 76 zero-loss dummy slots to reach 664 steps on 48 GPUs.

## What packing would change

Whole short examples could share a packed forward while retaining their own prefixes, attention boundaries, restarted positions, masks, and selected-frame subsets. Keeping these original examples and frames preserves epoch coverage. Joining them with unrestricted attention would change context. Cutting an example or treating its suffix as a new conversation would also change context and target boundaries.

A strict **7,500 total-position** budget cannot fit an existing full-length example plus its prefix and delay. The budget must distinguish conversation frames from total positions. Packing also changes examples per optimizer update and per-step frame denominators; the plan, step count, and learning-rate schedule must be reconsidered against the desired two-epoch exposure. Equal epoch coverage alone does not imply identical optimization. This audit implements no such change.

The key source rules are [row length construction](/home/param/pose-repo/finetune/duplex_distill/production/data_aloe_v2.py:110), [pose positions and rejection](/home/param/pose-repo/finetune/duplex_distill/production/data_aloe_posedup.py:164), [speech/text selection](/home/param/pose-repo/finetune/duplex_distill/production/lib/data_s3q.py:309), and [current normalization](/home/param/pose-repo/finetune/duplex_distill/runtime.py:108).
