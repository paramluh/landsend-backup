# Shared 10% decoder-frame and text-head profiles

**Mean step wall time fell from 23.8362 to 16.6987 seconds: 7.1374 seconds less, or 29.94%.** The same number of examples per step gives a measured throughput ratio of **1.4274×**. The main change is the user-requested reduction in pose higher-codebook training frames: pose now uses the existing approximately 10% speech frame sample. This is a sampling change, not a claim that the old full-frame objective computes faster unchanged.

A subsequent full-48-GPU run with the same shared sample and the profile-only **512-position text chunks / direct-addmm** candidate reached **15.0181 seconds per step**: another **10.06%** reduction, or **36.99%** below the original run. Its measured comparison and numerical limits are recorded below.

## Run and comparison scope

The original and shared-10% runs each executed six fresh-initialization steps on all **48 GPUs** across landsend-1,2,3,4,5,12. Both used the full architecture, production student and two teacher checkpoints, microbatch one, global batch 48, and per-layer depth activation checkpointing. Primary means use **steps 3–6: 192 rank-step records** in each run; all 288 records remain available. These short runs do not establish long-run throughput.

The shared-10% run completed all six steps on every rank and all six launchers exited zero. The **10:51:15 UTC / 03:51:15 PDT completion audit** verified every GPU idle, all completed event records reporting no teacher gradients, finite KL/data-CE values, and native-empty A conditioning for the pose teacher. That idle observation describes this run's completion, not later benchmarks. No production run resumed; no W&B logging, evaluation, or checkpoints were performed. The student remains trainable; teachers and Voxtral remain frozen. Dataset CE is a detached diagnostic in the tested implementation, and does not contribute to the training objective.

The exact plan digest remains `2275f1eab21f3326`. Every matched rank/step retains its host, text-position count, speech-depth count, conditioning metadata, and missing-pose status. Profile records store counts, not selected-index hashes: exact speech/pose frame identity additionally rests on the shared-index data implementation and passing CPU tests on real full-length, short, and missing-pose rows.

- [Machine-readable comparison](/home/param/pose-repo/finetune/duplex_distill/benchmarks/20261029/shared10/timing-summary.json)
- [Shared-10% raw profile](/home/param/pose-runs/duplex-distill-shared10-profile-20261029/profile-all-ranks.json)
- [Resolved configuration](/home/param/pose-runs/duplex-distill-shared10-profile-20261029/config.json)
- [Completion audit](/home/param/pose-runs/duplex-distill-shared10-profile-20261029/verification/completion-audit.json)
- [Original profile report](/home/param/pose-runs/duplex-distill-profile-20261029/PROFILE_REPORT.md)

## Where the time changed

Values are mean seconds over the same 192 rank-steps. CUDA event intervals include stream waits, communication waiting, and CPU launch gaps; they are not kernel-only measurements. FSDP communication remains included in forward/backward. Nested and CPU dispatch intervals must not be added to their parent intervals.

| Non-overlapping parent stage | Original pose 100% | Shared 10% |
| --- | ---: | ---: |
| Data-loader wait | 0.0983 | 0.0948 |
| Gradient reset | 0.0043 | 0.0043 |
| Frozen Voxtral encoding | 0.3027 | 0.3051 |
| Input preparation + count reduction | 0.9265 | 0.9895 |
| Frozen audio teacher forward | 1.9313 | 1.9007 |
| Frozen pose teacher forward | 1.9581 | 1.9984 |
| Student backbone + towers + text/cb0 KL | 3.7948 | 3.7748 |
| Speech depth: teacher + student + backward | 0.6869 | 0.6804 |
| Pose depth: teacher + student + backward | 5.6052 | 0.5796 |
| Decoder reduction, clipping, AdamW | 2.7969 | 0.7192 |
| Student backbone/tower backward | 5.4243 | 5.4007 |
| Metric reduction and processing | 0.0038 | 0.0025 |
| Main gradient clipping | 0.1105 | 0.0566 |
| Gradient checks | 0.0116 | 0.0118 |
| Main AdamW | 0.0303 | 0.0268 |
| Sum of parent spans | 23.6854 | 16.5452 |
| Unattributed wall gap | 0.1508 | 0.1535 |
| **Step wall time** | **23.8362** | **16.6987** |

“Student backbone + towers + text/cb0 KL” includes input embeddings and the Voxtral adapter, the backbone, text head and its custom gradient precomputation, both speech and pose towers, the pose-backbone link, and both codebook-0 KL calculations. The two higher-codebook depth decoders and later backbone/tower backward are separate rows.

Pose depth time fell from **5.6052 to 0.5796 seconds**. The following decoder-optimizer parent fell from **2.7969 to 0.7192 seconds**; its first gradient-all-reduce interval fell from **2.7584 to 0.6816 seconds**. These collective intervals include arrival imbalance, particularly ranks missing pose, and should not be reported as pure network transfer costs. Speech depth time stayed close: **0.6869 versus 0.6804 seconds**. The student text-head nested interval remains **1.3790 seconds** in this shared-sampling run.

## Both depth decoders now see the same sampled frames where pose exists

Across all six steps, **243** rank-steps have pose and their selected-frame counts exactly match speech; **45** lack pose and retain zero pose frames. In the steady analysis window, these counts are **164 valid / 28 missing**.

Across all 192 steady records, mean speech depth coverage stays **607.505 frames**; pose falls from **5,268.818 to 527.359 frames**. On the 164 pose-valid records alone, both decoders see **617.396 frames** on average and take **0.6910 seconds speech / 0.6786 seconds pose**. Thus their measured times are comparable when their work is comparable. On missing-pose records, pose depth is effectively zero; those ranks still participate in the required collectives.

| Nested depth work, averaged over all 192 records | Speech seconds | Pose seconds |
| --- | ---: | ---: |
| Frozen depth teacher forward | 0.1349 | 0.1124 |
| Student depth forward | 0.1371 | 0.1180 |
| Depth backward, including checkpoint recomputation | 0.4059 | 0.3475 |

Sampling changes only higher-codebook decoder indices and their frame-count normalization. The original text positions, conditioning, full speech/pose histories, and all valid codebook-0 targets remain. Pose remains absent for missing tracks. Evaluation still selects all valid depth frames before its existing 2,048-frame cap. The balanced objective remains one-third each text/speech/pose, with each media term using 0.25 cb0 plus 0.75 mean cb1–31. Losses or gradients need not equal the previous 100%-pose run.

## The preparation interval was predominantly waiting

The new nested timing isolates **985.153 milliseconds** of count-all-reduce stream time inside **989.510 milliseconds** of total preparation: **99.56%**. The reduction contains just five FP64 counts. All three role-array transfer intervals together average **3.084 milliseconds**; CPU count construction averages **0.169 milliseconds** of host dispatch.

| Nested preparation stage | Current-stream ms | CPU dispatch ms |
| --- | ---: | ---: |
| CPU masks/count construction | 0.159 | 0.169 |
| Five-count CPU→GPU transfer | 0.094 | 0.102 |
| Count all-reduce / stream waiting | 985.153 | 0.190 |
| Student arrays CPU→GPU | 1.351 | 986.316 |
| Pose-teacher arrays CPU→GPU | 1.099 | 1.112 |
| Audio-teacher arrays CPU→GPU | 0.633 | 0.644 |
| Student weights/positions | 0.381 | 0.393 |
| Pose-teacher weights/positions | 0.256 | 0.266 |
| Audio-teacher weights/positions | 0.209 | 0.218 |

The student-array CPU dispatch span is approximately **986 milliseconds**, while that transfer's current-stream interval is **1.351 milliseconds**. The host blocks there on earlier collective work; interpreting that host interval as a 986-millisecond data copy would be incorrect. CPU and GPU timelines place the same dependency wait in different named spans.

Arrival alignment supports the interpretation. On step 4, rank 0 spends **1.6877 seconds** in audio encoding and just **0.311 milliseconds** in the count reduction; faster encoders wait there instead. Across ranks, cumulative data-wait/reset/encode/prepare time finishes within **3.5 milliseconds** on that step. Steps 3, 5, and 6 show similarly aligned finishes. Audio encoding still includes waveform transfer, GPU resampling/log-mel, Voxtral, and its finite-value assertion; this profile has not separated those internals or proven the cause of occasional short-row encoding spikes.

Steady peak allocated memory ranges from **63.0617 to 64.7601 GB/GPU**, effectively unchanged from the original profile. Fewer sequential depth chunks reduce work, but do not remove the models, optimizer states, or peak per-chunk activations. Further performance candidates require their own measured comparison; no synthetic text-head speedup is extrapolated into these totals.

## Measured text-head candidate: 512 positions and direct addmm

This second candidate ran the full architecture on all 48 GPUs for six scratch steps with the shared-10% sample, identical plan, and the same original student/teacher initialization. Its configuration explicitly records `profiling_text.chunk_size=512`, `direct_addmm=true`, and `scope=profile_only`. Every rank-step matches the shared-10% baseline's text positions, speech/pose frame counts, and conditioning metadata. The completion audit verifies all ranks finished, all six launchers exited zero, finite metrics, no teacher gradients, matching source/configuration hashes across nodes, and no checkpoint or production resume.

Steps 3–6 average **15.0181 seconds**, compared with **16.6987** for shared sampling alone: **1.6807 seconds saved**, **10.06% less wall time**, or **1.1119×** the measured throughput. Relative to the original 100%-pose profile, this combination saves **8.8181 seconds / 36.99%**, with **1.5872×** measured throughput. These are observed whole-step results, not extrapolations from the earlier isolated text benchmark.

| Non-overlapping parent stage | Shared 10% seconds | Shared 10% + text candidate seconds |
| --- | ---: | ---: |
| Data-loader wait | 0.0948 | 0.0966 |
| Gradient reset | 0.0043 | 0.0043 |
| Frozen Voxtral encoding | 0.3051 | 0.3054 |
| Input preparation + count reduction | 0.9895 | 0.9783 |
| Frozen audio teacher forward | 1.9007 | 1.9046 |
| Frozen pose teacher forward | 1.9984 | 2.1185 |
| Student backbone + towers + text/cb0 KL | 3.7748 | 2.1651 |
| Speech depth: teacher + student + backward | 0.6804 | 0.6766 |
| Pose depth: teacher + student + backward | 0.5796 | 0.5774 |
| Decoder reduction, clipping, AdamW | 0.7192 | 0.6113 |
| Student backbone/tower backward | 5.4007 | 5.4850 |
| Metric reduction and processing | 0.0025 | 0.0024 |
| Main gradient clipping | 0.0566 | 0.0472 |
| Gradient checks | 0.0118 | 0.0113 |
| Main AdamW | 0.0268 | 0.0234 |
| Sum of parent spans | 16.5452 | 15.0075 |
| Unattributed wall gap | 0.1535 | 0.0106 |
| **Step wall time** | **16.6987** | **15.0181** |

The nested student text-head interval falls from **1.3790 to 0.2620 seconds**. The complete student forward parent falls from **3.7748 to 2.1651 seconds**, including its two towers and codebook-0 KL. The speech/pose tower intervals also shift from **0.8005/0.3019 to 0.4498/0.2018 seconds**, despite their unchanged code. These spans include scheduling and dependency waits; the full parent reduction must not all be labeled text-kernel compute saved. The backbone nested interval stays close (**0.9972/1.0029 seconds**), as do both depth decoder stages. Peak allocated memory remains **64.7601 GB/GPU**.

The full-run scalar comparison against shared sampling alone is:

| Metric | Maximum absolute difference over steps 1–6 |
| --- | ---: |
| `train/loss` | 1.63536893e-05 |
| `train/text_loss` | 6.97144051e-06 |
| `train/audio_loss` | 1.11744155e-05 |
| `train/pose_loss` | 4.46892294e-05 |
| `data_loss/loss` | 6.87176806e-05 |

The largest aggregate KL difference is **0.0000163537**. All six steps are finite. This is evidence of close scalar behavior in this short run, not elementwise gradient identity or bitwise equivalence: changing chunk sizes and matrix accumulation order changes floating-point rounding. No long-run quality or convergence conclusion follows from six steps.

- [Text-candidate raw records](/home/param/pose-runs/duplex-distill-shared10-fasttext-profile-20261029/profile-all-ranks.json)
- [Text-candidate configuration](/home/param/pose-runs/duplex-distill-shared10-fasttext-profile-20261029/config.json)
- [Text-candidate completion audit](/home/param/pose-runs/duplex-distill-shared10-fasttext-profile-20261029/verification/completion-audit.json)
