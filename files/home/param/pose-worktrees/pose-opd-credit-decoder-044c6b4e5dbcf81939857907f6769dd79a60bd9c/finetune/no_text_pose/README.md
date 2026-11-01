# No-text audio-and-pose fine-tune

This experiment removes text conditioning and text supervision from the Xabi
monoplex recipe. It starts from the **8,1,1 pretraining checkpoint at step 17850**,
the same initialization used by the production Xabi fine-tune. It does not start
from Xabi's final export or either transcript-conditioned fine-tune.

Implementation: [`../transcript_pose/train.py`](../transcript_pose/train.py),
`--recipe no-text`, with the adapter in
[`../transcript_pose/data_transcript_pose.py`](../transcript_pose/data_transcript_pose.py).
The hyperparameter source is
[`../xabi-main-17850/run.env`](../xabi-main-17850/run.env); the existing archived
trainer supplies the model, optimizer grouping and decoder updates.

## Experimental contract

| Item | Setting |
| --- | --- |
| Initialization | PPM 8,1,1 step 17850; fresh main and decoder optimizer states |
| Dataset | Exact compact Xabi/v2pure production cache; original train/eval split |
| Upfront transcript | None |
| Framewise text | Constant PAD in backbone and both media towers |
| Text objective | Disabled; no lexical, PAD, NEW_WORD or EOB targets |
| Audio/pose objectives | Original Xabi losses, stream coefficients 1 and 1 |
| Global batch | Original 64 logical rows per outer step; final partial batch retained |
| Hardware | landsend-1 through landsend-4, eight GPUs each, world size 32 |
| Preferred layout | Two logical rows per rank per forward; accumulation 1 |
| Memory fallback | One logical row per rank per forward; accumulation 2 |
| Duration | 200 outer optimizer steps; one pass over the source training clips |
| Depth decoders | Eight joint updates of each decoder per outer step |
| Logging | W&B project `param-pose-llm`; separate run name and directory |

The resulting model must also receive PAD as its framewise text input during
inference. There is no trained text output objective; a caller should emit PAD
in the text lane instead of sampling the old vocabulary head. Removing a text
loss alone would not enforce this inference behavior.

## What “no text” removes and retains

The adapter first builds the original Xabi example, including its deterministic
voice draw, then replaces its text IDs. This keeps media codes, positions,
attention segments, channel-presence masks, media labels and conditioning length
unchanged.

All generation positions use PAD ID **151670**, including positions formerly
containing wordpieces, NEW_WORD or EOB. Prefix wordpieces are also replaced with
PAD, including the ordinary `system` token and newline tokens. The original
voice-reference lane is already blank in this production recipe; the replacement
also prevents any lexical or timing tokens from surviving there.

Only the following structural tokens may remain, and only inside the existing
conditioning prefix:

| IDs | Purpose |
| --- | --- |
| 151644 / 151645 | Existing system-block start/end delimiters |
| 151676 / 151677 | Channel-A voice-block start/end delimiters |
| 151678 / 151679 | Channel-B voice-block start/end delimiters |

Channel A remains absent in the monoplex data, so its voice delimiters ordinarily
do not occur. Retaining these boundary embeddings preserves the structure of the
voice-conditioning block; they do not carry a transcript. No SOP/EOP transcript
block is added. Source examples have no persona payload.

Voice selection is unchanged: a deterministic hash selects a nominal one-third
of rows for a voice reference; a usable later same-speaker chunk must exist in
the recording's second half. Requested reference length is 38–125 codec frames
(3.04–10 seconds), with the original coverage/feasibility selection and fallback.
Thus the realized voice-conditioned fraction can be below one-third. The
experiment does not force an eight-second voice reference onto every example;
that was the controlled inference comparison's protocol.

Earlier ground-truth audio and pose codes remain teacher-forced during training,
with the production **two-frame media delay**. Each codec has 32 codebooks of
2,048 symbols at 12.5 Hz. No media are decoded, re-encoded or normalized again.
The archived training model has no separate `inp.g0` input; frame-zero geometry
is already part of the cached pose representation's preprocessing. This recipe
does not add a new geometry-conditioning module.

## Loss and optimizer accounting

For modality `m` (audio or pose), let `CE_m,0` be the mean cross-entropy of its
codebook-zero labels, and let `CE_m,1:31` be cross-entropy averaged over its valid
frames and all 31 residual codebooks. The reported aggregate is

```text
audio_loss = CE_audio,0 + CE_audio,1:31
pose_loss  = CE_pose,0  + CE_pose,1:31
loss       = audio_loss + pose_loss
```

Audio and pose have equal coefficients. Codebook zero has the same aggregate
coefficient as the entire residual-codebook mean: one residual codebook therefore
has **1/31** of codebook zero's coefficient. This preserves production Xabi's
weighting; it is not an equal-weight sum over all 32 codebooks.

Main-model CB0 gradients are normalized by each modality's valid target count
over the entire global step, accounting for FSDP's rank averaging. Residual
frames from every rank and every accumulation round are jointly partitioned into
eight decoder updates. Each decoder update uses its global partition's frame
count times 31 as the denominator. Its input gradients are weighted back by that
partition's share of the global frame count before propagation into the towers
and backbone. The existing pose decoder's backbone connection remains enabled
with coefficient 1; the unused old direct pose head remains unbuilt.

There are **200 main updates and 1,600 updates of each depth decoder**. Switching
from two rows in one forward to two accumulation rounds does not double the
number of decoder updates. The trainer retains the production two-pass update:
first collect detached tower states, update the decoders, then repeat the main
forward with gradients and propagate decoder-input gradients.

Changing world size and packing changes floating-point reduction order and the
rank-seeded decoder frame partitions. The objective, global row membership and
update counts match the production recipe; a bitwise identical optimization
trajectory is not claimed.

Peak learning rates and gradient-norm clipping:

| Parameter group | Peak LR | Clip norm |
| --- | --- | --- |
| Backbone / text embeddings / final normalization | `3e-6` | `0.5` |
| Special text outputs | `6e-5` | `1` |
| Text projection | `3e-5` | `1` |
| Audio and pose input code embeddings | `6e-5` | `1` |
| Speech tower | `1.5e-5` | `1` |
| Pose tower | `1.5e-5` | `1` |
| Pose decoder's backbone connection | `1e-5` | `1` |
| Speech depth decoder | `1.5e-5` | `1` |
| Pose depth decoder | `2e-5` | `1` |

Unused text-output parameters receive no text-objective gradients. The backbone,
its PAD embeddings and the shared final normalization still receive media-loss
gradients. The optimizer grouping is retained rather than pruning the model.

All optimizers use fused AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight
decay. The original schedule has seven warmup steps and linear decay to zero at
the final step. Decoder schedules use the same shape in decoder-update units
(56 warmup updates, 1,600 total). Decoder seed is `20261011`; dynamic/even decoder
minibatching and whole-decoder compilation in `default` mode remain enabled.
Decoder minibatching retains the production 300-frame nominal setting and
470-frame maximum setting. Compute is BF16 with FP32 parameters/optimizer state.

## Row packing and memory fit

The original **7,500 limit counts media frames per logical row**, not the entire
conditioning-plus-generation sequence. Several short clips can occupy one
logical row, each with an independent causal attention segment and restarted
position IDs. Conditioning and delay slots give a conservative maximum of 9,502
total positions per logical row.

The no-text plan preserves the original deterministic permutation and each
optimizer step's 64-row membership. With 32 ranks, the preferred layout packs two
logical rows into one variable-length forward, with a 19,004-position upper
bound. The fallback runs the same rows separately in two accumulation rounds,
with at most 9,502 positions per forward. Neither mode crops or splits source
clips to force conditioning into a 7,500-total-position budget.

The reconstructed cache plan contains **12,791 logical rows and 92,132,614 media
frames**, all presented once in 200 steps. The last step has 55 rows; absent
slots carry zero weight. The archived environment's header says 12,781 rows;
the executable plan and its digest are authoritative. The measured maximum
two-row token upper bound is 18,977, below the conservative 19,004 cap.

The separate `--benchmark-stress` mode repeats an actual logical row containing
7,500 media frames across multiple independent clips, and appends independent,
unsupervised padding to test the conservative position bound. It runs the complete
main update and both eight-update decoders. This tests the maximum total frame
and position counts, rather than the longest individual attention segment.
That synthetic stress plan is restricted to benchmarking and must never be used
for training; the actual run uses the unchanged full-epoch membership above.

## Data and reproducibility pins

| Artifact | Revision / preprocessing |
| --- | --- |
| `InternalCan/xabi-v2pure-production-cache` | `d4a80a57562c4d161744705a0435e372cf373d0c` |
| `InternalCan/ppm-811-17850-base-bf16` | `ccfd985473d52169e59aa4210aae50ac4e11531d` |
| `InternalCan/pose-tokenizer-32cb-200k-noshift` | `1d6ff4f0de89408cb054da618e309d4334596252` |
| Pose confidence preprocessing | `confidence-hold-03-frame0-init-v2` |
| Pose coordinate transform | `frame0-root-frame0-horizontal-shoulder-width-v1` |
| Mimi checkpoint | `89091b3e466eb6a9d11e537bf26b144f194978f7` |
| Mimi attention policy | `checkpoint_sliding_window_v1` / `mimi-local-attention-v1` |

The compact cache's historical SAM `text.npy` is not an input to this experiment.
The original PPM example builder still reconstructs words for deterministic
voice selection and provenance, then removes text before the model sees it.
The pose/Mimi tokenizers remain fixed; only the PPM model is fine-tuned.

## Launch, logging and checkpoints

From the CPU controller on landsend-0, after the assigned training GPUs are free:

```bash
finetune/transcript_pose/launch_cluster.sh \
  --run-dir /home/param/pose-runs/no-text-audio-pose-20261028 \
  --recipe no-text --mode train --resume never \
  --micro-batch-rows 2 --gradient-accumulation 1
```

Use `--micro-batch-rows 1 --gradient-accumulation 2` if the full stress update
does not fit. Use a separate directory for each benchmark and the actual run.
Only landsend-1 through landsend-4 participate in GPU training.

W&B records aggregate audio/pose losses, each of the 32 codebook losses per
modality, learning rates, gradient norms and evaluation losses. `train/loss` is
the sum of the two media metrics above; there is no text-loss contribution.
Run-local `config.json`, `plan.json`, `metrics.jsonl` and `wandb-url` preserve the
resolved recipe and its data-plan digest. System metrics and console capture
remain disabled as in the existing transcript recipes.

Evaluation runs initially, every 25 steps, and at weight-export checkpoints.
Weights are exported at **33, 67, 100, 133, 167 and 200**, matching Xabi's export
frequency. The local trainer additionally saves full resumable state at those
points and retains the latest two full saves, including main/decoder optimizer
state, scheduler and update counters. This is the existing local checkpoint
policy; the historical job disabled periodic and final full-state saves, while
retaining its save-on-stop path. Its scheduled exports were weights only.

```bash
touch /absolute/run/directory/control/STOP_AFTER_SAVE
```

Create this file on any participating worker to save weights and full state and
exit after the current outer update. Remove the request before resuming with
`--resume auto`. Resume requires the same four-node rank layout and data plan;
full state is distributed across all four nodes. `LATEST` points to each node's
latest completed full save, and completion markers are written after all ranks
finish.

The old expensive packed-loss GPU gate remains disabled in this local adapter,
as in the prior transcript experiments. Dedicated CPU audits check original
media/conditioning preservation, text removal, row coverage and batch membership;
the separate stress benchmark checks the complete training update and memory fit.
This document describes the recipe; the run directory's receipts establish which
layout was actually launched and whether it completed.

## Recorded launch

The run launched on **2026-10-28 at 23:23:45 UTC**, from source commit
`2ee719b54b6cefe0aae73855e7451cfe9c9ac676`, and completed its first training
update at **23:25:10 UTC**. The selected layout is **MB2 / accumulation 1** on
all 32 GPUs. No accumulation fallback was needed.

The full-frame fit check ran one warmup and one measured update, including both
eight-update depth decoders. The measured update took **20.24 seconds**, with
**56.52 GiB** maximum allocated memory across ranks. The actual training run's
second update took **19.96 seconds**; checkpoint and evaluation overhead are
additional. The initial total runtime estimate is 75–90 minutes.

W&B: [no-text-audio-pose-20261028](https://forge.coreweave.com/wandb/canopy-labs/param-pose-llm/runs/a5334709).
The run directory is `/home/param/pose-runs/no-text-audio-pose-20261028` on each
worker; the controller also has `launch-receipt.json`, `config.json` and
`wandb-url`. Benchmark receipts are in the sibling directory
`no-text-mb2-fit-20261028`. A launch receipt does not imply training completion;
check the run's `TRAINING_COMPLETE` marker and final checkpoint manifests.
