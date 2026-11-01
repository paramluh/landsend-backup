# Full-GT single-target distillation with nine references

This trainer restores the recovered full-GT experiment's single-target
construction and adds the nine-reference bank and the resolver objective. It
trains one dense student and one dense critic. Both critic objectives use
endpoint-prediction FM; both student objectives use the existing normalized DMD
surrogate. The teacher is a frozen copy of the configured nine-reference model.

The entry point is `learn_dmd_full_gt.py`, the configuration is
`experiments/distillation/full-gt-nine-ref.toml`, and the launch wrapper is
`run_dmd_full_gt.sh`. The entry point selects the full-GT mode of the shared
training lifecycle in `learn_dmd_joint.py`; it does not select the four-target
generated-history experiment. The previous joint entry point remains available
with its existing semantics and an incompatible resume identity.

**Use an equal average of stabilizer and resolver: coefficients `.5` and `.5`
for both critic FM and student DMD.** There is no tilt by diffusion-interval
width or measured gradient magnitude, and no automatic gradient balancing.
The checked-in `objective_weights` table supplies these selected values.

The FM scale probe exercised both scoring objectives and their parameter
gradients on 20 microbatches of two samples per objective. It does not validate
the complete student DMD update, distributed optimizer, checkpoint/resume
lifecycle, throughput, or convergence of this new training mode.

## Data and model inputs

Each row contains eight temporal positions:

```text
sink | GT f−6, f−5, f−4, f−3, f−2, f−1 | target f
```

The nine reference latents are loaded separately from the actor bank, with
their own validity mask. They are not inserted into those eight temporal
positions and are not selected from the target clip's temporal frames. The
reference patch embedding and attention/RoPE conventions are inherited from the
nine-reference model implementation. The supplied sink is the clip's frame zero.
Pose and audio remain aligned to each selected source timestamp.

The default assets are the existing studio-only CSV, pose-v3 latents, local Mimi
codes/frontend, validated studio reference bank, and step10424 EMA teacher
already provisioned for the earlier nine-reference run. Verify their paths on
the new training node. Do not substitute app data or node-local CSV shards.
All ranks must read the same full-corpus CSV and reference configuration.
Reference dropout is disabled by default; unavailable bank properties retain
the existing invalid-reference/pad semantics.

`eligibility_history=6` makes source frame seven the first eligible target.
Every subsequent eligible frame supplies a single-target candidate, so adjacent
rows overlap in their GT context. The recovered opposite-limits comparison
required sixteen available predecessors to match its no-GT arm's eligibility.
That matching restriction is not needed here: there is no sixteen-frame
generation or burn-in in this full-GT trainer. Setting
`eligibility_history=16` would restore the earlier target eligibility while
still loading only six GT predecessors.

## Stabilizer generation and globally fixed context

One target starts from Gaussian noise at `t=1`. It takes eight differentiable
Euler updates, two per stabilizer band, ending at `h=.2`:

```text
1 → .9 → .8 → .7 → .6 → .5 → .4 → .3 → .2
```

Only this target is integrated. Other present-slot frames are supplied GT
conditioning; their velocity outputs are discarded. There are no generated
prefix frames or gradient paths through other generated targets.

For each of the six GT predecessor identities, sample one noise tensor `eta_i`
and define its entire conditioning path as

```text
C_i(tau) = (1 − tau) * GT_i + tau * eta_i.
```

Keep that noise tensor fixed across every target integration step and the later
teacher/critic scoring queries. Reconstruct a context state directly from this
formula when its slot time changes. Do not resample noise each step and do not
integrate a predecessor with the model's velocity. History slots receive
GT-derived values at `.2`; earlier present slots receive values at the lower
times required by the target's current band.

The target moves through the following window:

| Target slot | Three history identities at `.2` | Earlier present identities |
|---|---|---|
| 3, highest noise | `f−6, f−5, f−4` | `f−3, f−2, f−1` at `t−.6, t−.4, t−.2` |
| 2 | `f−5, f−4, f−3` | `f−2, f−1` at `t−.4, t−.2` |
| 1 | `f−4, f−3, f−2` | `f−1` at `t−.2` |
| 0, lowest noise | `f−3, f−2, f−1` | None |

The predecessor latent values are fixed, detached inputs. Compute the sink and
all six fixed `.2` history representations once per stabilizer rollout and
reuse their per-layer K/V, selecting the three histories required by each band.
Build this cache in the enclosing gradient mode: during student generation it
remains connected to autograd, so parameters that process context still receive
gradients. Earlier present-slot conditioning changes with diffusion time and
is recomputed each integration step. Reference K/V caching remains enabled and
also preserves its gradient connections. Scoring for the student DMD update is
under `no_grad`; critic scoring for FM training is differentiable with respect
to the critic. Caches are local to a model invocation and are never reused
across optimizer updates or different models.

### History dropout

Use the recovered full-GT run's independent drop probabilities, indexed by
distance before the supervised target:

| Predecessor | Drop probability |
|---|---:|
| `f−1`, nearest | 0% |
| `f−2` | 10% |
| `f−3` | 20% |
| `f−4` | 20% |
| `f−5` | 0% |
| `f−6` | 0% |

One sampled validity decision belongs to each predecessor identity. It follows
that identity as the moving window changes; it is not a new decision per
attention slot or integration step. The same retained context and mask are
used during student generation and its teacher and critic scoring. Dropped
frames remain dropped in both history and earlier present positions, with
invalid/pad semantics, zero time, suppressed pose, and suppressed present audio.
The sink, target, and reference bank are not part of this temporal dropout mask.
The dataloader does not apply another temporal dropout policy.

The stabilizer and resolver samples use the same dropout policy but draw their
own masks, even when paired by source target ID. "Globally fixed" means a
sample's predecessor identity has the same validity in all of its rollout
steps and model queries; it does not mean freezing one mask across objectives,
distinct examples, or future optimizer rounds.

## Resolver generation

The same single-target row supplies the resolver objective. Noise its GT target
to `.2` with the ordinary clean-endpoint path, then take one differentiable
Euler step to zero:

```text
x_.2 = .8 * GT_target + .2 * epsilon
y_res = x_.2 − .2 * v_student(x_.2, .2, conditioning).
```

The resolver receives the sink, nine references, and its last two GT
predecessors **clean at zero**. Those two predecessors use the final two
identity decisions from the full-GT dropout policy: the older one drops with
10% probability and the nearest one is retained. Its mask is fixed during
generation and teacher/critic scoring. There is no additional parent-window
categorical dropout draw, no generated history, and no resolver CFG.

## Critic endpoint FM and student DMD

After generating a stabilizer endpoint `y` at `h=.2`, draw one independent
scoring time uniformly within `(.2,1)`, excluding only `1e-6` endpoint guards,
and draw fresh target-corruption noise. This target noise is independent of the
fixed predecessor noises. Corrupt the generated endpoint with

```text
a = (1 − t) / (1 − h)
s² = t² − h² * a²
x_t = a * y + s * epsilon
u_t = a' * y + s' * epsilon.
```

Place this one noisy target in the present slot appropriate to its sampled
time: slot zero for `(.2,.4]`, one for `(.4,.6]`, two for `(.6,.8]`, and three
for `(.8,1)`. Earlier present slots and the three history slots use the retained
GT context at their appropriate times; future slots are invalid. The teacher
and critic receive identical target corruption, context noises, validity mask,
pose, audio, sink, and reference inputs.

The stabilizer critic loss is

```text
k_h(t) = (1 − h) * s² / t
FM_stabilizer = mean_CHW((k_h(t) * (v_critic − u_t))²).
```

The implementation forms the finite residual algebraically instead of
subtracting a large unweighted derivative target near `.2`.

For resolver critic training, the generated endpoint is at **zero**, even
though its one-step student generation began at `.2`. Draw a scoring time in
`[.02,.2]` and fresh target noise, use `x_t=(1−t)y+t*epsilon`, and train with

```text
u_t = epsilon − y
FM_resolver = mean_CHW((t * (v_critic − u_t))²).
```

Thus both FM terms measure squared endpoint-prediction error. Raw resolver
velocity MSE is not used as its training objective. The two endpoint objectives
receive equal coefficients despite their different measured gradient scales.
Keep the intrinsic `k_h(t)^2` scaling in each endpoint objective; equal outer
coefficients do not remove or compensate for it.

Both student objectives use the usual per-target normalized DMD surrogate:

```text
g = (v_teacher − v_critic) /
    max(mean_CHW(abs(v_teacher − u_t)), normalizer_epsilon)
L_DMD = .5 * mean_CHW((y − stop_gradient(y − g))²).
```

The scoring models, corruption path, and normalizer are detached in this student
update. Gradients reach the student through the generated endpoint. Do not add
the FM `k²` factor to DMD: its velocity-to-endpoint conversion cancels between
the numerator and normalizer, apart from the stated numerical floor.

The selected teacher guidance scale is `2.5`. Replace only the teacher velocity
in DMD with `v_null + 2.5 * (v_conditional - v_null)` when the sampled stabilizer
time lies in the highest-noise band `(.8,1]` (the first denoising stage, target
slot 3 in the packed layout). Exact band boundaries belong to the lower band.
All other stabilizer bands and the resolver use the conditional teacher.
The critic always predicts the conditional student score; neither critic nor
student receives CFG. The current no-crossbleed training CSV has 14,414 chunks
and zero bad-audio flags, so all student, critic, and conditional-teacher rows
receive full audio. There is no random audio dropout in this trainer.

For a teacher microbatch of `B` rows, pack the `B` conditional rows and only the
`K` highest-noise null copies into one `B+K` forward. Each null copy drops audio
for the **whole clip**, including every earlier present-frame audio window;
its target noise/time, pose, references, sink, predecessor noise, and keep mask
are identical to its conditional partner. Every rank makes one reference
prefill and one main forward even when `K=0`, keeping node-sharded teacher
weight collectives aligned. For the selected two-row student microbatch this
requires teacher batch shapes 2, 3, and 4. Compiler limits are 256 per code
object and 512 accumulated, with failure on exhaustion rather than fallback.

## Batches, epoch, and optimization

Each optimizer update accumulates 160 single-target stabilizer rows and 160
single-target resolver rows into the same model, then reduces gradients, clips,
and steps once. At 40 GPUs, each objective supplies four rows per GPU. Source
loading uses two-row microbatches; execution can regroup those rows separately
for each objective and model. The global batch must be divisible by
`world_size * micro_batch_size_per_gpu`; choose a compatible topology or local
microbatch size. Do not silently change the global batch.

An outer round has four critic optimizer updates followed by one student
optimizer update. The critic uses four distinct full batches of 160 source
rows, each paired across the two objectives. The student uses a fifth distinct
batch, also paired across objectives. The student stays fixed during all four
critic updates. Each critic sample is generated without gradients and scored
for critic FM with gradients. `rollout_pool_batch_size_per_gpu=16` combines all
four updates' rows into one detached stabilizer rollout and one detached
resolver rollout per GPU. Endpoints and every context tensor are then sliced
back into the four original updates, which remain sequential. Audio inputs are
prepared at the original source microbatch granularity before regrouping.
The later student batch is generated with gradients and receives DMD. Critic
samples are not reused four times.

`critic_stabilizer_micro_batch_size`, `critic_resolver_micro_batch_size`,
`student_stabilizer_micro_batch_size`, and `student_resolver_micro_batch_size`
control execution sizes without changing source IDs, the global batch, or
objective normalization. Each size must divide the four local objective rows.
The corresponding `<role>_<objective>_checkpoint` settings select `block` or
`none`, scoped across forward and backward while preserving shared compiled
functions. Optional `reference_checkpoint=block` checkpoints only the nine-frame
reference prefill independently of the target policy; reference K/V gradients
remain connected. This permits testing target forwards without checkpointing
while avoiding the reference prefill's larger activation footprint. Omit the
setting to use each objective's policy for both prefill and targets.
`full_gt_trim=false` keeps all four present slots during stabilizer
rollouts to reduce shape specialization; future slots remain invalid and
causally inaccessible. This trades additional arithmetic for fewer graphs.

The deterministic critic schedule selects successive circular blocks after
the current student block in the shuffled corpus. Critic rows may recur in
other rounds; within a round, all four critic batches and the student batch
have disjoint target IDs. This requires at least 800 eligible rows with the
default batch/update counts; startup rejects a corpus too small to satisfy
that contract.

One epoch means that every eligible source target contributes exactly once to
each **student** objective. The two objectives share source IDs and reference
draws; generated target noises are separate. The number of student updates is
`ceil(eligible_targets / 160)`. Extra critic batches do not advance the student
epoch. The final student batch uses zero-weight padding rather than dropping
or repeating targets, and each objective is normalized by its actual global
valid-row count. Critic batches remain full. `student_rows_seen` counts unique
paired source targets, not twice that number and not critic rows.

The current run sets `max_student_steps=8000`: it follows the first 8,000
student batches of this unchanged epoch plan, then stops. The full corpus has
168,284 student batches; this run is deliberately shorter than one full epoch.
The dataset, target eligibility, shuffle, and per-round source identities are
unchanged by the cap.

The initial 150 critic-only rounds are separate from the student epoch. The
student schedule contains the configured 8,000 updates; the critic schedule includes
warmup and all four updates in each subsequent round. Both use AdamW with
peak LR `2e-6`, betas `(.9,.99)`, weight decay `1e-4`, and clipping at 2. Student
LR warmup is 100 updates and critic LR warmup is 200 updates, followed by cosine
decay to 10% of peak over their respective derived schedules. The shared
lifecycle shortens LR warmup for sufficiently small corpora.

The selected balanced objectives are

```text
L_critic  = .5 * mean(FM_stabilizer)  + .5 * mean(FM_resolver)
L_student = .5 * mean(DMD_stabilizer) + .5 * mean(DMD_resolver).
```

Apply each `.5` coefficient once after normalizing that objective over its
actual global valid-row count. Do not divide by two again or by accumulated
microbatch count. This is an arithmetic average, not a sum with coefficients
`1` and `1`; the absolute scale affects total gradients and clipping. Do not
apply the earlier `.8/.2` interval weighting or compensate for unequal gradient
norms. The endpoint factors and per-target DMD normalizers remain as defined
above. Equal objective coefficients do not imply equal gradient contributions.

The corrected GT-input probe found median GT velocity-FM gradient norms of
`.442` (stabilizer) and `.505` (resolver), and endpoint-FM norms of `.131` and
`.0116`, respectively. These are unweighted parameter L2 norms of two-sample
batch-mean losses, before clipping, over 20 microbatches per objective. The
selected coefficients are a training decision, not a fitted normalization of
those measurements. Probe records and histograms are in
`/mnt/paramfs/param/outputs/fm-scale-probe-gt-baseline-20261019`.

## Checkpoints, logging, and sensitivity

The implemented metric names, formulas, and evaluation protocol are specified in
[metrics-and-evaluation.md](metrics-and-evaluation.md). CPU acceptance checks
cover reductions, protocol behavior, media construction, and publication payloads;
full model evaluation and live W&B delivery still require runtime validation.

The full-GT lifecycle saves every 800 student updates, on manual requests, and
at the final planned student update. The initial critic-only rounds do not
advance this cadence: the first scheduled save is student update 800, total
round 950, and the final save is student update 8,000, total round 8,150.
Every save includes raw student weights, raw critic
weights, student EMA weights (decay `.999`), exact optimizer/scheduler/RNG and
cursor state, and completion metadata. Teacher weights are not saved. Retention
keeps the latest eight completed weight/evaluation sets and optimizer state
only for the latest resumable checkpoint. Partial saves are not promoted to
completed checkpoints. Save cadence and other operational settings do not
change the training resume identity; coefficients and training semantics do.

Evaluate every **400 student updates** independently of the **800-update
checkpoint cadence**. The headstart does not advance either counter. At a
checkpoint, final-save, or manual-save boundary, run its evaluation once and
skip the separate in-memory evaluation. Intermediate evaluations use the
current raw student, plus the existing optional EMA policy, without writing
weights or optimizer shards. Every rank's training RNG is restored after the
probe and publication, including failures. Local receipts and console events
record probe, tracking, and total elapsed seconds; these are not extra W&B
training scalars.

Intermediate reports use a fresh evaluation UUID with
`evaluation_kind=in_memory` and `checkpoint_id=null`, with student/round counters,
source configuration, and the last durable checkpoint recorded separately.
The W&B captions and gallery identify these as in-memory evaluations. Their
receipts live under `evaluation-runs/`; artifacts remain under
`evaluations/<evaluation-id>/`. They do not update `latest.json` or enter
checkpoint retention. A failed intermediate probe leaves an incomplete
receipt; after process failure its lost in-memory model cannot be recovered
from that receipt. Resume uses the latest durable checkpoint and creates a
fresh evaluation UUID when training reaches the cadence again. Saved-checkpoint
evaluation recovery retains its existing behavior. The legacy W&B
`eval/checkpoint_step` axis continues to count total rounds for comparability.

One explicit exception supports the authorized teacher guidance transition
from `1.0` to `2.5`: pass `resume_allow_teacher_guidance_change=true` together
with `resume_from`. All other training/data/model/topology/schedule fields must
still match, and both optimizers, FP32 masters, EMA, RNG, and counters are
restored normally. A run-local `resume-transitions.json` receipt records the
exact old/new signatures and source checkpoint identity after state restoration;
resume runtime metadata links this receipt. The original epoch plan remains
unchanged. Retention recognizes only that enrolled transition when keeping
eight weight sets and the latest optimizer state across the guidance boundary.
Without this explicit opt-in, changing guidance remains incompatible.

The run directory must be on a filesystem shared by every rank. Each rank
writes its own optimizer/master/FP32 EMA/RNG shard concurrently, flushes it,
and atomically replaces its temporary file. Rank zero verifies all expected
shard sizes before publishing checkpoint completion. A shard write or flush
failure is reported to every rank and prevents promotion and pruning of
previous checkpoints. Raw student/critic weights and the merged BF16 EMA export
remain rank-zero writes. The rank-state format and existing resume reader are
unchanged; training remains paused until the save and evaluation finish.

Online W&B publishes the fixed 54-scalar training contract, with one observation
per outer round on `train/step`. System statistics and legacy progress, timing,
framewise, and sensitivity scalar namespaces are disabled. Local `metrics.jsonl`
retains operational telemetry plus globally reduced measurement sums/counts.
Diagnostics add no model forwards. Teacher and critic statistics reuse their
existing scoring outputs; `student_rollout_rms` observes the already-computed
student target velocities across solver steps. This is a sampler statistic,
not a same-input student/teacher comparison. The metric contract is
`full-gt-training-v2`; the previous extra-forward `student_rms` series is retired.

Each checkpoint evaluates four stabilizer target slots and one resolver target
with paired endpoint perturbations, replaying generated baseline context.
Plots show the actual three/two predecessors, sink, and valid-reference mean.
A separate 189-latent, 753-video-frame rollout uses the same student's stabilizer
and one-step resolver, without temporal GT feedback. The pinned Wan2.2 VAE
decodes generation and reconstructed GT. Evaluation includes precompression RGB
L2, AutoBench background drift curves and four summaries, audio/video playback,
and lossless RGB originals. Raw weights are the default; `eval_weights='both'`
also evaluates EMA. EMA is saved in either case.

Artifacts live under `RUN_DIR/evaluations/<checkpoint-id>/<weights>/`; checkpoint
directories remain flat for retention. A run-specific local gallery preserves
all evaluation rounds independently of weight retention. Cloudflare deployment
awaits the user's VM details. Evaluation freezes source/configuration/input
hashes in `evaluation-protocol.json` and uses `eval/checkpoint_step` in W&B.
Its special layouts temporarily use eager blocks, restoring the shared training
compile wrapper afterward.

## Launch and handoff

The configuration selects `.5` for both objectives and the rebuilt strict bank.

```bash
: "${MASTER_ADDR:?Set the rendezvous host address}"
: "${NODE_RANK:?Set this node's distinct zero-based rank}"

NNODES=5 SLOTS=8 NODE_RANK="$NODE_RANK" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT=29932 \
  ./run_dmd_full_gt.sh experiments/distillation/full-gt-nine-ref.toml
```

Run one wrapper per node with the same configuration and coefficients. The
wrapper defaults to `.venv/bin/torchrun`;
`TORCHRUN` can identify another prepared environment. Set NCCL/Gloo network
interfaces for the cluster if required. The wrapper does not SSH, create
credentials, kill jobs, or provision assets. Its default port and local
compiler-cache directories differ from the old joint launcher.

Verify the common studio CSV, reference bank validation, teacher checkpoint,
Mimi assets, W&B authentication, and output filesystem before an eventual GPU
launch. The configured fresh output root is
`/mnt/paramfs/param/outputs/full-gt-nine-ref/training`. Use the first saved
`epoch-plan.json`, runtime metadata, and metrics to inspect the actual eligible
target count, schedule lengths, batch counts, and losses. The implementation
does not establish a successful launch or a throughput estimate by itself.

To request a durable save and continue, touch `RUN_DIR/save`; to save and exit,
touch `RUN_DIR/save_q`. Exact resume passes
`--set 'resume_from="/absolute/completed/checkpoint/directory"'` together with
the same chosen coefficients and training configuration. The new full-GT
identity rejects checkpoints from the earlier generated-history joint run.
Resume restores optimizer, EMA, RNG, schedules, and the student epoch cursor;
it does not silently migrate a different experiment's state.

Implementation ownership:

- `distillation/full_gt_data.py`: single-target full-epoch plan, separate critic
  groups, shared bank loading, and canonical inference-evaluation rows.
- `distillation/full_gt_dmd.py`: fixed GT context/noise/masks, moving target
  slot, single-target integration/scoring, and resolver conditioning.
- `learn_dmd_full_gt.py`: dedicated mode-selection entry point.
- `learn_dmd_joint.py`: shared optimizer, lifecycle, tracking, and full-GT
  objective orchestration.
- `distillation/training_metrics.py` and `training_tracking.py`: fixed scalar
  contract, weighted accumulators, and W&B scalar/media publication.
- `distillation/endpoint_sensitivity.py`, `rollout_evaluation.py`, and
  `training_evaluation.py`: paired probes, student AR video, drift/GT curves,
  pinned checkpoint evaluation, and the local gallery.
- `distillation/checkpoint_retention.py`: exact resume identity and retained
  checkpoint/optimizer generations.
