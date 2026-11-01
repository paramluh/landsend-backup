# Two-teacher pose OPD with finite causal credit

This package trains the pose-duplex student on its own text, audio and pose
rollouts over the **first two minutes of each row**. The frozen pose-monoplex teacher scores pose distributions; the
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

The selected `conversation_prefix_frames=1500` keeps the first 120 seconds at
12.5 Hz. It is a deterministic prefix, with one example per source orientation
per epoch: no random crop, later window, or additional example is created.
The selected `exclude_short_rows=true` excludes source rows shorter than 1500
frames before the epoch plan is built. All **13,566 eligible orientations**
in the audited selected dataset already contain at least 1500 frames, so this
rule adds no exclusions and every selected conversation has the same duration.
Other recipes can retain shorter rows with `exclude_short_rows=false`.
`conversation_prefix_frames=null` retains
complete stored rows for other recipes. Source duration remains in
`original_conversation_frames`; metadata records both original and selected
frame totals.

Each selected prefix is one continuous cached rollout across its exchanges.
EOB and silence do not terminate it. There is no ground-truth reset between turns.
The real A waveform is fixed and limited to the same prefix. Future B dataset text/audio/pose and labels
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

The selected pose teacher uses **CFG strength 3** for all 32 codebooks. Its
conditional and unconditional passes read the same student text and pose
history, including each actual sampled pose-codebook prefix. The unconditional
pass replaces valid Mimi audio with the native encoded digital-silence sequence,
including reference and prefill audio; absent audio remains absent. Text and
pose are preserved. Each teacher categorical distribution is
`softmax(logits_unconditional + 3 * (logits_conditional - logits_unconditional))`.
The silence artifact is SHA256-pinned in the recipe. Both the direct KL and its
causal score credit use this guided distribution. Student sampling has no CFG.

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
Student sampling has no CFG, grammar filtering, top-k/top-p or repetition penalty. This is
not a probit-link distribution. Every sampled action records its behavior
log-probability; replay compares these with its differentiable log-probabilities.
Cached and full-sequence BF16 kernels can differ numerically, so the trainer
logs discrepancies rather than claiming bitwise actor/replay identity.
`--profile-policy-agreement` adds bounded per-stream/codebook sampled-logprob
statistics and full categorical actor/replay KL probes at identical histories,
including early and final sampled positions. Sampled-logprob discrepancies and
categorical KL are different quantities; neither alone proves gradient parity.

The selected actor uses inverse-CDF sampling from those same full-vocabulary
probabilities. FP64 cumulative sums and independent FP64 uniforms avoid the
loss of small-probability outcomes that a FP32 CDF can cause. Uniforms are
prepared once per row and RNG stream, outside the captured frame graphs.
This changes seeded trajectories relative to `torch.multinomial`, while retaining
the categorical distribution; the sampler backend is part of the run identity.
The native multinomial backend remains available for comparisons. Depth KV
caches hold the 32 positions actually used by the 32-codebook decoder. The
selected actor uses FP32 probability-times-value accumulation to match the
learner depth attention's actual precision, plus the `tuned_m16` linear backend.
The selected GPU paths completed the 40-GPU batch-320 benchmark. Numerical
actor/replay differences remain and are quantified in [BENCHMARKS.md](BENCHMARKS.md);
the backend names do not imply bitwise equivalence.

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
admitted clip counts in dataset metadata. Source train/eval split membership is
retained. An optional `max_row_frames` capacity limit checks the selected
conversation length after the explicit prefix rule. The selected epoch contains
**452.2 oriented row-hours**, including the two-second prefill in each example.

## Execution and memory

All trainable student parameters are FP32 FSDP shards, with BF16 computation and
FP32 gradient reduction. Frozen teachers use BF16 parameter storage, retaining
the audio adapter's trained gain in FP32. Every pair of local GPUs hosts an
ephemeral BF16 cached actor, refreshed from the current student using
cast-before-all-gather publication. The selected `rollout_batch_per_rank=8`
generates **16 concurrent rows per GPU pair**, independently of the learner
microbatch. Each row has separate RNG streams, prefix and audio inputs. Each
learner forward packs `physical_microbatch` selected row prefixes with isolated attention
segments and causal-credit row IDs. Gradient accumulation reaches
`effective_batch_per_rank` before one optimizer step. All microbatches use the
same published student and whole-update reduction denominators. The target is
**eight rows per rank on 40 GPUs**, arranged as **20 two-GPU actor teams**, or
**320 rows per update**. The selected student physical batch is **8**, so each rank processes its full
effective batch in one replay. This completed the batch-320 GPU benchmark. Smaller physical batches can accumulate to the same
effective eight rows. There are **43 optimizer updates per epoch**, or **129 over three epochs**:
each epoch has 42 full batches followed by 126 real rows and 194 zero-weight
dummy slots. Epoch shuffles use distinct epoch seeds; actor sampling uses the
absolute update, rank and row slot, so it does not repeat across epochs.
Actor weights/caches are released
before learner activations are created. No stale-policy queue is used.
`prefetch_rows=true` loads one future effective batch in a persistent spawned
CPU-only process during the current update; the initial batch overlaps model
loading. This isolates the archived loader's process-wide import/environment
changes. The saved row order, resume cursor and dummy mapping are retained.
Frozen GPU audio encoding uses the same validated native implementation.
The selected actor split places 30 backbone layers on the first GPU and 34 on
the second to balance memory. The selected frozen audio encoder stays resident, avoiding CPU/GPU transfers
between updates. All ranks release unused publication buffers before actor KV
allocation. CPU parking remains configurable for larger-memory recipes.

`fixed_sequence_frames=1636` pads each learner/teacher row to the maximum native
prefix length 134, plus 1500 conversation frames and two media-delay positions.
Padding is appended after the native timeline: prefix locations and rotary
positions of real tokens remain unchanged. All padding loss and credit masks
are false, and causal attention prevents padding from influencing earlier
outputs. Row boundaries remain isolated. Eight rows therefore occupy 13,088
packed positions.

Frozen teacher batches are independent of student microbatches. The selected
audio teacher scores eight rows together. Pose CFG packs eight conditional and
eight unconditional rows into one **16-segment, 26,176-position** temporal
forward, with isolated attention boundaries. Its depth scoring also batches
the two CFG branches. Teacher outputs are prepared once for the effective
update and reused by every student accumulation step or benchmark candidate.

The learner first computes detached KL values for every generated text position
and all 32 audio and pose codebooks. It then computes causal credit coefficients
and replays the student with direct KL plus credit. Text heads use bounded
vocabulary chunks. Media depth chunks backpropagate immediately into detached
tower/link roots; their accumulated gradients enter one final temporal backward.
All ranks execute matching padded chunk calls, with zero weights on padding.
Decoder residency avoids repeated parameter gathers. The selected
`decoder_gradient_sync="per_replay"` accumulates full FP32 decoder gradients and
synchronizes on the globally aligned final chunk, trading extra GPU memory for
fewer collectives. The selected `checkpoint_depth_decoder=false` removes depth-layer
rematerialization; **backbone and tower activation checkpointing remain enabled**.
Decoder chunks contain 256 frames and text chunks 128 positions. The conservative
configuration defaults (`per_chunk`, depth checkpointing enabled) remain available.
Exactly one optimizer step follows all losses. See [BENCHMARKS.md](BENCHMARKS.md)
for the matched-trajectory batch comparison and maximum-shape checks.
The selected `teacher_depth_cache="cpu"` stores exact FP32 teacher logits in
pinned host memory between scoring and replay. This avoids repeating audio and
CFG pose depth forwards and their parameter gathers; no logits are quantized.
Eight selected two-minute rows require approximately **5.6 GiB per rank** for
the two media caches alone (`8 × 1475 × 31 × 2048 × 4 × 2` bytes).
Set the option to `"none"` to trade that memory for recomputation.

A CUDA microbenchmark rejected plain `torch.compile` for depth RMSNorm: BF16
Q/K forward and input-gradient relative RMS differences were approximately
0.00166 and 0.00284, including with precision-cast emulation enabled. Although
that isolated helper ran about twice as fast, it did not preserve the required
numerics. Production retains the existing normalization implementation.

The selected loss weights are **1/3 per modality**, with **0.5 for cb0 and
0.5/31 for each of cb1..31** in both media branches. Loss normalization is
explicit: `row_mean` averages modality distributions
within each row and then averages real rows; `position_mean` uses global valid
position counts. Codebook weights and the three modality weights are separately
configurable. Dummy tail slots have zero loss, credit and denominator weight.

The selected optimizer is AdamW at **3e-6**. The starting settings retain betas
`(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, global gradient clipping `1.0`,
three epochs and Amu's OPD learning-rate schedule: seven warmup updates, a
plateau at 3e-6, then thirteen cooldown updates, with start/end LR 3e-7. The
schedule follows the absolute update and does not restart at epoch boundaries
or resume. Checkpoints are saved every **60 updates** and at the final update
(**60, 120, 129** for this run); only the latest optimizer state is retained,
while model checkpoints remain available. Configuration, source
hashes, data plan, sampling policy and model identities are recorded. Resume
requires the same identities, world size and host/rank mapping. Model, optimizer,
RNG and next-update cursor use the existing node-local resumable checkpoint
implementation. No model/evaluation/upload job is started by preparation.

## Live monitoring and four samples per update

The selected `rollout_samples_per_update=4` archives the first four real rank-zero
rows after each completed rollout, without consuming training RNG. The complete
three-epoch plan has four real rows in those slots on every update. Exact source
A waveforms, sampled text/audio/pose arrays, behavior log-probabilities, masks,
conditioning and row identities are saved under `raw-rollouts/update-XXXXXX/`.
Each update directory is published atomically only after all four payloads exist.
A repeated update after resume preserves its previous snapshot as history.

`samples_decode.py` is an independent CPU worker: it decodes each entire Mimi
conversation including prefill, decodes the matching 32-codebook pose tokenizer,
and publishes the website's four-sample manifest after all assets are complete.
The pose tokenizer represents normalized **2D** skeletons; the viewer receives
an explicitly labeled planar embedding with z=0 and confidence stored separately.
The two-second prefill and 120-second audio/pose timeline remain aligned.
The trainer does not wait for decoding and no training GPU is used by the worker.

The live website consumes a read-only allowlist mirror of rank-zero telemetry
and these raw snapshots. Existing raw modality/codebook KL, combined gradient
norm, learning rate, progress and timing are connected. Conditional lexical/
category KL and separately measured direct/credit component gradient norms
remain unavailable; the website does not invent those values.

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

# Run on each of five allocated eight-GPU hosts. Set NODE_RANK and MASTER_ADDR
# for that job. The same source, recipe and inputs must be present on every host.
LAUNCH=( "$PYTHON" -m torch.distributed.run --nnodes=5 --node-rank="$NODE_RANK"
  --master-addr="$MASTER_ADDR" --master-port=29500 --nproc-per-node=8
  -m finetune.pose_opd_credit.train --config "$RECIPE" )

# Production: no synthetic inputs, cache reuse, or optimizer warmup.
"${LAUNCH[@]}" --run-dir /absolute/path/run

# Real two-minute rollouts, one unsaved update, with steady-state Adam memory.
# Save completed CPU trajectories for subsequent learner-only measurements.
"${LAUNCH[@]}" --run-dir /absolute/path/benchmark-real \
  --max-updates 1 --no-save --warm-optimizer \
  --profile-actor-graphs --profile-replay-stages --profile-policy-agreement \
  --benchmark-save-rollouts /absolute/path/rollout-cache

# Reuse those identical trajectories and weights for batches 2, 4 and 8.
# Retain --profile-policy-agreement when loading a cache saved with its probes.
"${LAUNCH[@]}" --run-dir /absolute/path/benchmark-replay \
  --max-updates 1 --no-save --warm-optimizer --profile-replay-stages \
  --profile-policy-agreement --benchmark-load-rollouts /absolute/path/rollout-cache \
  --benchmark-replay-microbatches 2,4,8

# Stress the maximum selected real A/conditioning shape without waiting for
# actor generation. Synthetic B tokens cycle the forced prefill; this measures
# shape/memory/compute only, with no on-policy or generated-quality claim.
"${LAUNCH[@]}" --run-dir /absolute/path/benchmark-shape \
  --max-updates 1 --no-save --warm-optimizer --benchmark-shape-only \
  --profile-replay-stages --benchmark-replay-variants \
  '[{"physical_microbatch":8,"checkpoint_depth_decoder":true,"decoder_gradient_sync":"per_chunk"},{"physical_microbatch":8,"checkpoint_depth_decoder":false,"decoder_gradient_sync":"per_replay"}]'

# CPU mathematics, dataset, actor-publication and orchestration checks.
CUDA_VISIBLE_DEVICES='' $PYTHON -m unittest discover \
  -s finetune/pose_opd_credit/tests -v
```

`--benchmark-replay-microbatches` and `--benchmark-replay-variants` reuse the
same completed bundles, teacher outputs and model weights. Each candidate starts
with zero student gradients; only the final candidate reaches the optimizer
step. Variants may change `physical_microbatch`, `decoder_chunk_frames`,
`text_chunk_frames`, `decoder_gradient_sync`, and `checkpoint_depth_decoder`.
OOM is fatal; the process does not attempt to recover partially executed FSDP
collectives. These comparisons and rollout-cache options require one bounded,
unsaved update and prohibit resume.

Rollout caches store CPU seed rows, exact audio features, generated tokens,
behavior likelihoods and any policy probes. Per-rank SHA256 receipts are agreed
across all ranks before an atomic `COMPLETE` manifest is installed on each node.
Loading checks source, model, dataset, plan, prefix and actor-policy identity;
only the explicitly supported learner scheduling settings may differ. It skips
the audio frontend and actor. Synthetic and real-actor caches have distinct
identities and cannot be substituted. Use a fresh run directory for each test.

`--warm-optimizer` allocates Adam moments through a zero-gradient, zero-decay step
that leaves weights unchanged. It advances Adam's counter and therefore changes
subsequent bias correction; it is for memory/timing benchmarks, never the normal
production initialization. `--benchmark-prefix-frames 750` can further shorten
a real diagnostic rollout to its first minute under bounded, unsaved guards;
it does not change the production two-minute recipe. The older
`--benchmark-microbatches` option changes physical batch across successive real
updates, regenerating trajectories each time, so use the replay options for a
controlled comparison.

`--profile-replay-stages` reports host submission spans and current-stream CUDA
event intervals for teacher preparation, student scoring, depth backward and
final temporal backward. It adds one collection synchronization per replay,
not one per chunk. Overlapping FSDP communication is not isolated by these
timings. Gradient norm/sum/max checksums help compare candidates but do not prove
elementwise equality.

Tests include exhaustive expected-gradient enumeration of a branching policy,
codebook14 causality, analytic text-head/dense gradient parity, actor sampling
and publication, real-data prefix alignment, available/absent conditioning
clips, fixed-padding masks, and actual two-rank FSDP replay. They compare
independent teacher batches with packed CFG, student accumulation, deferred
synchronization and dense references. Disabling depth checkpointing has a CPU
output/gradient parity check. CPU tests do not establish GPU kernel parity,
production peak memory or throughput. Different BF16 batch shapes, reduction
orders and actor kernels can cause numerical differences even when the
objective and causal graph match; report measured tolerances and categorical
agreement rather than claiming bitwise model-gradient identity.

Full models require dedicated multi-host resources until capacity is profiled;
the example does not establish that eight GPUs can hold the FP32 student,
optimizer, two teachers and cached actor. The source models' 64-rank training
layout is historical context, not a measured requirement for this method.
