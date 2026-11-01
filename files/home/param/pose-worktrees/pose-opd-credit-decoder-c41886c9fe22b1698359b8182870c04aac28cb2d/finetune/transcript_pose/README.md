# Transcript-conditioned audio and pose

Two recipes share the same production cache and step-17850 initialization:

| CLI recipe | Transcript prefix | Framewise loss | Micros per GPU |
| --- | --- | --- | --- |
| `aligned-media` (default) | Full frame-aligned word/PAD/NEW_WORD lane | Audio + pose | 2 |
| `compact-all` | Ordinary tokenized transcript without timing tokens | Text + audio + pose | 1 |

Both use 56 GPUs, 7500 total positions per packed micro, 200 outer steps,
original Xabi voice conditioning, eight updates of each depth decoder per outer
step. The aligned recipe follows Xabi's checkpoint schedule; the compact recipe
saves automatically only at the final step. Always use distinct run directories.

## Compact transcript with three supervised streams

For `--recipe compact-all`, cleaned source words are joined with ordinary spaces
and tokenized as one text string. The prefix is `SOP | transcript | EOP` and
contains no PAD or NEW_WORD timing tokens. It includes source words even when
the original timed word lane could not fit all their wordpieces. The original
framewise text/audio/pose sequence follows the prefix; only generation positions
have targets. Prefix tokens are conditioning only.

Each stream has coefficient 1. Text uses ordinary full-vocabulary token-mean CE:
the production text multiplier of 10 and lexical multiplier of 5 are both reset
to 1. Audio and pose retain their production definitions, each **cb0 mean CE +
mean CE over the 31 residual codebooks**. `train/loss` and `eval/loss` include all
three streams; `train/text_loss` and `eval/text_loss` are added to the existing
media/codebook/LR/gradient metrics. Batch size is 56 packed micros, with no
gradient accumulation beyond that one micro per rank.

Overlong clips retain original encoded media and framewise text when split. A
split's compact transcript includes complete words whose timestamps overlap it;
a word crossing a boundary appears in both prefixes. Total prefix, conditioning,
media and delay-tail length must fit the same 7500-position budget.

```bash
finetune/transcript_pose/launch_cluster.sh \
  --run-dir /home/param/pose-runs/compact-transcript-text-audio-pose-20261028 \
  --mode train --recipe compact-all --resume never --port 29465
```

This starts fresh from the pretraining base with fresh optimizer states, once
the preceding run has completed and the seven nodes are free.
Automatic full-state saves and BF16 weight exports happen only at step 200.
The manual `STOP_AFTER_SAVE` control still saves and exits at the next optimizer
boundary. Evaluation remains every 25 steps and at the end.

## Aligned transcript with media supervision

This recipe starts from the **811 pretraining checkpoint at step 17850**, the
same initialization as the production Xabi fine-tune. It reads the exact compact
production cache; the cache's SAM text IDs are ignored. Original PPM words,
tokenization, deterministic voice selection and media codes are reconstructed
using the archived production helpers.

Each independent attention segment contains:

```
Xabi conditioning | SOP | all frame-aligned text | EOP | text + audio + pose
```

The prefix copies the complete generation word lane, including PAD and NEW_WORD.
SOP/EOP (151680/151681) reuse existing trained delimiter embeddings. There is no
audio/pose input or prediction target inside the transcript prefix. The original
two-frame media delay and framewise text conditioning remain in the generation
part. Media generation can attend to the entire transcript. Packed clips have
separate causal attention segments and restart position IDs. Overlong clips are
split without re-encoding media or rebasing pose geometry.

The training loss is audio cb0 CE + mean audio residual-codebook CE + pose cb0 CE
+ mean pose residual-codebook CE. Text CE is skipped entirely; the backbone head
normalization still runs because the towers need it. Both replicated depth
decoders take the original **eight joint nanostep updates per outer step** across
both accumulated micros. FSDP gradients use the original global token counts.

`../xabi-main-17850/run.env` supplies the production learning rates, AdamW
settings, clipping, seven-step warmup, linear decay, decoder settings, voice
conditioning and evaluation cadence. The explicit changes are 56 GPUs, two
packed micros per rank, a **7500-position total** budget, transcript prefix and
zero text loss. There are 200 outer steps. The resulting deterministic plan
covers roughly 79.35% of the source training frames with no repetition; 30 clips
require splitting. This is not a full epoch.

Data: `InternalCan/xabi-v2pure-production-cache` at
`d4a80a57562c4d161744705a0435e372cf373d0c`, installed at
`/home/param/datasets/xabi-v2pure-cache` on every node.

Base: `InternalCan/ppm-811-17850-base-bf16` at
`ccfd985473d52169e59aa4210aae50ac4e11531d`, installed at
`/home/param/models/ppm-811-17850-base-bf16` on every node.

## Running

Source `environment/activate.sh`; `environment/bootstrap.sh` and its lockfile
document environment recreation. `launch_cluster.sh --run-dir /absolute/path
--mode benchmark-old` runs the original Xabi objective and original 64-row batch
membership on seven nodes. `--mode benchmark-new` measures the new recipe.
Both use one warmup and two measured steps by default, keeping the original
200-step LR schedule. Timing reports stay in local `benchmark.json`, excluding
startup, evaluation and checkpoint I/O. Use a different directory for each mode.

`--mode train` runs the full recipe and logs to W&B project `param-pose-llm`.
Only aggregate media losses, learning rates, 32 losses per modality, gradient
norms and aggregate evaluation losses are logged. Decoder gradient norms are the
maximum pre-clipping norm across their eight updates. System metrics and console
capture are disabled. The run URL is written to `wandb-url` in the run directory.

Full resumable state **and** BF16 weight exports are saved at steps
**33, 67, 100, 133, 167, 200** for `aligned-media`, and **200 only** for
`compact-all`. The most recent two full saves are retained;
weight exports are retained separately. Full saves contain FP32 model shards,
main and decoder AdamW states, scheduler and decoder-update counters. Completion
markers are written only after all ranks finish. State digests are checked when
resuming. `LATEST` points to the latest full save on each node.

```
touch /absolute/run/directory/control/STOP_AFTER_SAVE
```

On any participating node, this requests a full save and weight export after the
current outer optimizer step, then a clean exit. `control/STOPPED` records the
saved step on node 0. Before resuming, remove the request file from the node where
it was created, then launch the same run directory with `--resume auto` (the
launcher defaults to `never` for fresh runs).
Resuming requires the same nodes, rank layout and data plan. Saves are distributed
across node-local disks; all seven nodes' files are needed. Do not treat a single
node's export directory as a complete standalone model.
