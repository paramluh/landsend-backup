# Two-teacher pose OPD with finite causal credit

This package trains the pose-duplex student on its own complete text, audio and
pose rollouts. The frozen pose-monoplex teacher scores pose distributions; the
frozen audio-duplex teacher scores text and audio distributions. Every scored
distribution uses the full analytic `KL(student || teacher)`. There is no sampled
KL estimate, dataset cross-entropy, reward model, PPO clipping, GRPO baseline or
additional timing/preservation objective.

The existing `finetune/duplex_distill` production model, frozen Voxtral encoder,
data formats and checkpoint implementation are reused. Its teacher-forward
objective, 30% conditioning draw and 10% depth-frame sample are not reused.

## Objective and causal graph

Let `K_e` be the analytic categorical KL at decision `e`, and let `w_e` include
the configured modality/codebook and distributed reduction weights. The loss is

```text
direct = sum_e w_e K_e
credit_weight[a] = sum_{e: a is a sampled ancestor of e within the frame window}
                       stop_gradient(w_e K_e)
loss = direct + sum_a credit_weight[a] * (log_pi[a] - stop_gradient(log_pi[a]))
```

This is Amu's zero-valued, first-order score-function surrogate, applied to frame
windows and the actual duplex sampling graph. The reported scalar equals the
direct KL, while the credit term contributes gradients. The finite cutoff
intentionally omits older score factors; it is not claimed to be the complete
gradient of an unlimited-horizon expected objective.

At model frame `t`, text is sampled from the previous backbone state. Both media
towers then condition on that state, their respective previous codes, and the
current text. Audio and pose are independent branches within this frame. Each
branch samples cb0 through cb31 autoregressively. All three generated streams
then enter the backbone and influence subsequent frames.

For the pose cb14 KL, the same-frame ancestors are text and pose cb0..13. Audio
cb0..31 and pose cb14..31 are excluded. The cb14 draw does not determine its own
analytic KL, which sums the entire distribution before that draw. Prior-frame
audio, text and pose can all receive its credit. Earlier pose can also receive
credit from later text/audio KL, even though the audio teacher excludes pose:
the student's text/audio distributions depend on that earlier pose.

The selected lookback is **8 preceding frames** (640 ms), plus same-frame causal
ancestors. The sampled token at the scored KL's own slot is always excluded.
`lookback_frames` is measured on native model-frame positions. With
`window_mode="previous_frames"`, a KL at `t` credits frames `t-L..t-1` and the
current-frame causal prefix. With `"including_current"`, it credits
`t-L+1..t-1` and the current prefix. A frame is 80 ms. The cutoff limits sampled
action credit; deterministic transformer/KV computation remains differentiable
through the complete replay history. There is no truncated hidden-state BPTT.

`objectives.py` computes the coefficients with frame prefix sums and reverse
codebook sums. It does not construct an action-by-loss matrix.

## Rollouts, teachers and conditioning

One row is one continuous cached rollout, including every exchange. EOB and
silence do not terminate the row. There is no ground-truth reset between turns.
The A waveform is fixed for the row. Future B dataset text/audio/pose and labels
are erased before the actor receives the seed. The actor's sampled B arrays
become the histories and codebook prefixes used by all scoring/replay passes.

An available B **audio-only** conditioning clip is always used identically by
student and teachers. If no supported clip exists, all three models receive the
native empty five-token system prefix, with no reference audio, text or pose.
Available clips have their spoken words removed; native system/prefix boundary
tokens remain. Reference pose is absent. The separate two-second conversation
prefill still contains all three B streams.
Optional `voice` and `voice_pose` modes are available, but are not the selected
recipe. The monoplex teacher retains its native
empty A lane. The audio teacher sees the exact same A waveform/frozen Voxtral
latents as the student and uses its own adapter; it does not receive pose.

The initial two seconds are 25 physical frames. Text generation/scoring starts
at model frame 25. Because media have a two-frame delay, model frames 25 and 26
still force ground-truth audio/pose; media generation/scoring starts at model
frame 27. Forced prefix/prefill actions never receive score credit. Two final
forced PAD text positions flush the final media frames without scoring text.

Pose prefill uses the output speaker's recorded pose for those initial 25
physical frames. The current loader first requires the source pose track to
pass the dataset's validity checks, then verifies that the prefill has valid
codes. This is separate from voice-reference availability. Ground-truth pose
after the prefill is erased; generated pose is scored by the pose teacher.

Sampling is untruncated categorical softmax at temperature one, over all 151682
executable text IDs and all 2048 outcomes for each media codebook. Native
PAD/NEW_WORD/EOB output rows are retained. Padded text-head rows are excluded.
There is no CFG, grammar filtering, top-k/top-p or repetition penalty. This is
not a probit-link distribution. Every sampled action records its behavior
log-probability; replay compares these with its differentiable log-probabilities.
Cached and full-sequence BF16 kernels can differ numerically, so the trainer
logs discrepancies rather than claiming bitwise actor/replay identity.

The selected student is raw pose-duplex **export-424** (two epochs), the
pose teacher is raw bandfix monoplex **export-355**, and the audio teacher is
**raw-w14503/export-253**. [CHECKPOINTS.md](CHECKPOINTS.md) records pinned source
evidence and the audio teacher's explicit exported-bf16 gain policy; the student
retains its exact fp32 gain. JSON configurations name each checkpoint and data
variant explicitly. The adapter supports the installed
SAM-separated stores and a validated mapped raw-code variant. Raw B/reference
Mimi values are distinct from the native A waveform/gain path. The selected
recipe uses raw codes. An audit of all 23,554 mapped A entries found identical
original/remapped/installed waveform metadata and gains, with zero mismatches.
It selects later-sibling B voice clips with the original loader's supported
clip rule. The selected `missing_clip_policy="empty"` retains rows without a
reference, using the native empty prefix. `"skip"` retains the previous strict
clip selection, and `"error"` fails preparation on unavailable inputs.
Of 15898 candidate orientations, 2320 lack valid pose and 12 have invalid
frame-zero pose normalization, leaving **13566**: **6410 with a reference** and
**7156 without one**. Exclusions are recorded in `data-selection.json` and
admitted clip counts in dataset metadata. Rows retain
their complete stored duration; an optional capacity limit rejects overlong
rows rather than cropping them. Source train/eval split membership is retained.

## Execution and memory

All trainable student parameters are FP32 FSDP shards, with BF16 computation and
FP32 gradient reduction. Teachers are frozen. Every pair of local GPUs hosts an
ephemeral BF16 cached actor, refreshed from the current student using
cast-before-all-gather publication. The pair generates two independent rows
sequentially, then each rank trains on one row. Actor weights/caches are released
before learner activations are created. No stale-policy queue is used.

The learner first computes detached KL values for every generated text position
and all 32 audio and pose codebooks. It then computes causal credit coefficients
and replays the student with direct KL plus credit. Text heads use bounded
vocabulary chunks. Media depth chunks backpropagate immediately into detached
tower/link roots; their accumulated gradients enter one final temporal backward.
All ranks execute matching padded chunk calls, with zero weights on padding.
Decoder residency avoids repeated parameter gathers while preserving each
chunk's gradient reduction. Exactly one optimizer step follows all losses.

The selected loss weights are **1/3 per modality**, with **0.5 for cb0 and
0.5/31 for each of cb1..31** in both media branches. Loss normalization is
explicit: `row_mean` averages modality distributions
within each row and then averages real rows; `position_mean` uses global valid
position counts. Codebook weights and the three modality weights are separately
configurable. Dummy tail slots have zero loss, credit and denominator weight.

The selected optimizer is AdamW at **3e-6**. The starting settings retain betas
`(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, global gradient clipping `1.0`,
one epoch and checkpoints every two hours. Configuration, source
hashes, data plan, sampling policy and model identities are recorded. Resume
requires the same identities, world size and host/rank mapping. Model, optimizer,
RNG and next-update cursor use the existing node-local resumable checkpoint
implementation. No model/evaluation/upload job is started by preparation.

## Run and validate

Use the repository's existing training Python environment. The new worktree
does not require copying or modifying that environment. Set the same FA3/Triton
paths used by the existing trainer. The selected configuration is
`finetune/pose_opd_credit/recipe.raw.json`; its checkpoint paths, clip policy,
window convention, weights and reduction are explicit.
`--prepare-only` writes `preparation.json` with the complete identity, plan and
selection audit; it does not overwrite an existing training identity.

```bash
cd /home/param/pose-worktrees/pose-opd-credit
PYTHON=/home/param/pose-repo/.venv/bin/python
RECIPE=finetune/pose_opd_credit/recipe.raw.json

# Dataset/recipe validation only; no CUDA model allocation.
$PYTHON -m finetune.pose_opd_credit.train \
  --config "$RECIPE" --run-dir /absolute/path/run --prepare-only

# On each allocated host, set NNODES, NODE_RANK and MASTER_ADDR for that job.
# Eight local ranks per host; total host count must be chosen after profiling.
$PYTHON -m torch.distributed.run --nnodes="$NNODES" --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 --nproc-per-node=8 \
  -m finetune.pose_opd_credit.train \
  --config "$RECIPE" --run-dir /absolute/path/run

# Bound a real-model smoke run without writing model checkpoints.
# Append these to the same command: --max-updates 2 --no-save

# CPU mathematics, dataset, actor-publication and orchestration checks.
CUDA_VISIBLE_DEVICES='' $PYTHON -m unittest discover \
  -s finetune/pose_opd_credit/tests -v
```

Tests include exhaustive expected-gradient enumeration of a branching policy,
codebook14 causality, analytic text-head/dense gradient parity, actual cached
actor adapter methods, available and absent clips on installed production rows, prefill
delay masks, removal of future dataset B values, two-rank DTensor publication,
and chunked replay gradient comparisons. CPU tests do not establish production
GPU kernel parity, throughput or peak memory; run the bounded GPU smoke on the
selected hardware before treating those properties as validated.

Full models require dedicated multi-host resources until capacity is profiled;
the example does not establish that eight GPUs can hold the FP32 student,
optimizer, two teachers and cached actor. The source models' 64-rank training
layout is historical context, not a measured requirement for this method.
