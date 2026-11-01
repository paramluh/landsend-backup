# Nine-reference joint stabilizer/resolver distillation

This implements the October 17 request on `param/bimodal`, including the later
clarification that the **student uses six detached prefix frames**, while the
**critic sample generator stays at seven total frames**. The sink and nine
reference images remain supplied conditioning. Every other stabilizer image is
generated. The follow-up batch correction uses a full batch160 for each mode
every update, with explicit shared FM/DMD loss weights instead of a 3:1
microbatch schedule. The original October 17 implementation had static review
only. On October 18 the user authorized numerical validation, optimization,
five-node benchmarking and a production launch; the current work is described
below. A benchmark receipt or training log is required to establish runtime
success; the configuration alone is not evidence of a completed launch.

The entry point is `learn_dmd_joint.py`, with
`experiments/distillation/joint-nine-ref.toml` and `run_dmd_joint.sh`. It produces
one shared dense student and one shared dense critic for both objectives. The
teacher is a frozen copy of the supplied nine-reference checkpoint. There is no
LoRA, and only the student has EMA weights.

## The rollout and losses

The student stabilizer generates six prefix frames without gradients and four
targets with gradients. Every generated frame starts at `t=1` and receives eight
Euler updates, two per band, ending at `t=.2`. The four targets all receive DMD
losses; gradients through earlier generated target frames and their cached K/V
remain connected to later targets.

The six-frame prefix fills the first target's complete window:

```text
                         first gradient-tracked target enters
9 refs + sink | history 1,2,3 | present 4,5,6,7
               detached        detached 4,5,6; tracked 7
```

This is a continuous staggered wavefront. At that instant frames 1–3 have reached
`.2`, while frames 4–6 are still denoising in the lower present bands with
gradients disabled. They are not completed and then artificially renoised to
restart the wavefront. Subsequent tracked targets are frames 8–10. Prefix
generation begins with the normal startup pads, but the first tracked target
enters a fully populated temporal window. The six prefix frames' trajectories
and their cached conditioning are detached.

For critic training, the **student** generates three prefix frames plus four
targets: seven total. This entire operation is under `no_grad()`. The critic
then evaluates the generated samples with gradients. This deliberate difference
in prefix lengths follows the user's explicit correction and follow-up answer;
do not silently change the critic to ten frames. It also means the critic is
fitted on a shorter-startup distribution than the one producing the student's
DMD targets. That statistical approximation is part of this experiment; the
code does not claim the two generated context distributions are identical.

### Preserved history dropout

The old stabilizer upload used independent nearest-predecessor-first drop
probabilities `[0.0, 0.1, 0.2, 0.2]`, not the resolver's categorical rule.
The user explicitly selected the prefix-only adaptation for this joint rollout:
sample one mask over generated prefix identities relative to the first target,
keep it fixed through all target denoising steps and teacher/critic scoring,
and keep **all four supervised targets valid**, including as conditioning for
later targets. This does not introduce independent target-specific masks.

For the six-frame student prefix, the oldest-to-newest drop probabilities are
`[0, 0, .2, .2, .1, 0]`. For the three-frame critic prefix, only the available
three predecessors exist, giving `[.2, .1, 0]`; the fourth predecessor's `.2`
probability does not create an extra frame. Later targets retain all earlier
target conditioning and do not resample probabilities by distance.

Prefix trajectories themselves use ordinary undropped generation. Target
conditioning uses a separate masked view, including prefix frames still in
earlier present slots: those conditioning K/V are recomputed from the same
pre-update states under the mask. Reusing undropped present K/V would retain
hidden-state contributions from a dropped prefix frame. Dropped conditioning
uses pad validity, time zero, suppressed pose, and suppressed present audio.
These context forwards generate no additional temporal frames. Sensitivity
evaluation disables stochastic history dropout to isolate each perturbation.
As in the old bootstrap-then-mask method, prefix latent values were generated
with their full available past. Dropout changes the later conditioning view;
it does not regenerate those latents under a counterfactual missing history.

The four target endpoints are renoised into the four critic/teacher present
slots, using each slot's own band:

| Slot | Scoring interval | Teacher velocity |
|---|---|---|
| 0 | `.2 < t <= .4` | conditional |
| 1 | `.4 < t <= .6` | conditional |
| 2 | `.6 < t <= .8` | conditional |
| 3 | `.8 < t < 1` | audio CFG, `w=2.5` |

The last three generated prefix endpoints supply history at `.2`. All four
renoised targets are present together, with the usual causal attention plan.
The teacher's guided slot uses `v_null + 2.5*(v_cond-v_null)`. The null forward
drops audio for the **whole present window**, leaving pose, sink, references,
history, and the noisy target tensors unchanged. Slots 0–2 retain conditional
velocities. “First frame” here means the first/highest-noise denoising band,
slot 3, not the oldest chronological target. The student and critic have no
runtime guidance pass; the student learns the guided teacher field.

The stabilizer retains the endpoint-aware corruption kernel for `t0=.2`:

```text
a = (1-t)/(1-t0)
s² = t² - (t0*a)²
x_t = a*y + s*epsilon
v* = a'*y + s'*epsilon
```

The critic uses finite endpoint-prediction FM, `mean[k*(vF-v*)]²` with
`k=(1-t0)*s²/t`, computed through its nonsingular residual. The student uses the
existing normalized endpoint DMD surrogate, per target:

```text
g = (vT-vF) / max(mean_CHW(abs(vT-v*)), 1e-8)
L = 0.5 * mean_CHW((y-stop_gradient(y-g))²)
```

All scoring and normalization are detached during the student update. The final
mean includes all four target frames. This preserves the repository's normalized
endpoint weighting; it is not a claim to reproduce an unweighted KL gradient.

Resolver contributions retain the existing GT method: noise each GT target to
`.2`, take one differentiable Euler step to zero, score it against clean GT
history, use resolver scoring times `[.02,.2]`, and apply the existing clean-path
FM/DMD losses. GT versions of earlier targets provide later targets' resolver
history. There is no stabilizer-prefix generation or teacher CFG in this branch.
The prior parent-window categorical history dropout `[.1,.1,0]` is retained for
resolver samples only: 10% drop the oldest parent-history frame, 10% drop the
oldest two, and 80% keep all. This is not independent dropout on every resolver
history slot. The packed resolver does not read the oldest parent slot; the
effective behavior is that the first target's older history is dropped in 10%
of rows, with every other resolver-history input retained. This reproduces
`ResolverWindows._window` and the existing `resolver-gt.toml` configuration.
The same validity mask is used during generation and teacher/critic scoring.
References are added consistently to this existing resolver layout.

## Accumulation, batches, and one epoch

Every student optimizer step accumulates **160 stabilizer target frames and
160 resolver target frames**, followed by one gradient reduction, clipping,
AdamW step, and EMA update. Each critic optimizer step uses the same two full
batches and loss coefficients. `global_batch_size=160` is target frames **per
objective**, not windows or the sum of both objectives. Each canonical row has
four targets, so each objective uses 40 global rows and each update contains
320 target losses total.

The paired objectives use the same selected source-window IDs and reference
draws. Their generation noise is drawn separately, and resolver-only GT-history
dropout remains unchanged. At 40 GPUs with one row/GPU, this is one stabilizer
microbatch then one resolver microbatch. At 8 GPUs it is five stabilizer
microbatches then five resolver microbatches. The global row budget (40) must
be divisible by `world_size * micro_batch_size_per_gpu`; otherwise startup
fails rather than silently changing batch size. There is no optimizer step or
gradient clearing between the two objectives.

Both critic FM and student DMD use the same weighted mean:

```text
L_critic  = 0.8 * mean(FM_stabilizer)  + 0.2 * mean(FM_resolver)
L_student = 0.8 * mean(DMD_stabilizer) + 0.2 * mean(DMD_resolver)
```

The weights follow the repository's per-present-slot FM convention. The
nine-reference universal configuration `experiments/studio/aloe-studio-v2.toml`
uses five present slots. `UniversalWidePipeline.training_t` assigns them equal
bands over `(0,1]`, and `models/pose/loss.py` averages valid present-slot MSE
equally. The lowest slot covers `(0,.2]`; the other four cover `(.2,1]`.
Grouping these as resolver and stabilizer gives `1/5` and `4/5`. The separate
bimodal trainer agrees: `learn_bimodal.py` explicitly computes
`num_present / sum(num_present)` using four stabilizer slots and one resolver
slot. Its two-mode trainer rejects reference conditioning, so it is supporting
evidence rather than proof of the nine-reference checkpoint's exact lineage.
Packing four independent
resolver targets into a distillation row changes the batch layout, not the
parent objective's relative importance. The ratio is applied explicitly once;
there is no further division by two or by the number of accumulated microbatches.
`objective_weights` is the single shared setting for both FM and DMD. Metrics
report the unweighted mode means and their weighted total separately.

Each round first generates one detached pool for the two full objective batches.
The four critic optimizer updates reuse that fixed pool with fresh scoring
times and fresh corruption noise. The student does not change between those
critic updates. The subsequent student update generates fresh samples from the
same source windows, using its six-frame stabilizer prefix. This avoids four
extra prefix generations and does not consume another data epoch for the critic.

The optional 150-round critic warmup is separate from the student's epoch. It
may reuse windows and does not advance the student data cursor. It uses the
same full paired batches, shared loss weights, and four critic updates per round.

`joint_data.py` deterministically shuffles all eligible nonoverlapping
four-target windows once. A source clip of usable length `L` contributes
`max(0, floor((L-7)/4))` windows, with target starts 7,11,15,…, matching the
existing GT resolver eligibility. The sink, first six context positions, and
an incomplete final group of fewer than four frames are not target windows.
"One epoch" means every such eligible window appears in the **student
stabilizer loss once and student resolver loss once**. `student_rows_seen`
counts unique paired source rows, not the sum across modes. The number of
student updates is `ceil(eligible_rows / 40)`; LR decay is derived from this
revised epoch length.

Every rank must see the same full-corpus CSV and have access to all listed clips
and the actor bank. The trainer rejects `node_sharded=true`. Use shared data or
replicate the full corpus with identical paths; do not supply different local
CSV shards. This intentionally avoids the old shortest-node truncation.

The last group uses zero-weight padding when fewer than 40 source windows
remain. This is the only smaller effective batch: it avoids dropping or
repeating examples to finish the epoch. Both modes have the same actual valid
count, and each mode's mean is normalized by its own global valid-row count
before its `.8` or `.2` coefficient is applied. The 4:1 importance ratio therefore
also holds in the tail. All ranks execute the same forwards/backwards, including
zero-weight tail microbatches. The CSV, ordered source mapping, shuffle, epoch,
reference configuration, topology, global batch size, and objective weights
are included in the resume identity. The new paired plan rejects checkpoints
from the earlier 3:1 microbatch plan; it cannot silently resume their cursor or LR.

Student LR warmup is 100 optimizer updates; critic LR warmup is 200 critic
updates. Each then follows cosine decay to 10% of the peak LR `2e-6`, with the
length derived from the actual epoch plan. The student schedule excludes critic
warmup; the critic schedule includes it. Warmup is shortened for very small
corpora. A one-update degenerate schedule uses the existing schedule helper's
single peak-LR update, since there is no interval over which to decay.

## Checkpoints, EMA, and sensitivity

Save every **1,000 total training rounds**, counting critic warmup, and at the
final epoch round even if it is not divisible by 1,000. Explicit `save` and
`save_q` requests remain independent of that periodic interval. Cadence is an
operational setting excluded from resume compatibility; changing it does not
change optimizer/scheduler steps, data order, or the total epoch length. Both
old and current compatibility identities are normalized in code, without
rewriting saved checkpoint metadata. Every completed checkpoint contains:

- `student.safetensors`: raw shared student;
- `model_ema.safetensors`: shared student EMA, default decay `.999` per student
  optimizer update, initialized from the student and unchanged during warmup;
- `critic.safetensors`: raw shared critic;
- `rank-00000.pt` etc.: both optimizers' FP32 masters/moments, schedulers, EMA
  shadows, RNG state, and phase/epoch counters;
- `metadata.json` and `checkpoint.complete.json` identifying the completed save.

Every rank writes its own optimizer-state file concurrently to the shared
output directory. Raw student/critic weights and merged EMA are written by rank
zero. Each rank flushes its shard and reports success; rank zero verifies all
expected shard sizes before committing completion. The directory must be
accessible to every rank and needs enough disk for the new checkpoint **while the
previous optimizer state still exists**; the trainer checks capacity before
saving. Only after a new checkpoint is complete are older optimizer shards in
that run removed. The authorized production policy keeps the latest **eight
completed checkpoints**, including raw/EMA/critic weights and evaluation
receipts, with optimizer/RNG state retained only for the newest checkpoint.
Retention also runs at resume startup. Partial saves use unique staging
directories and do not collide with later retries.

Sensitivity runs at this same save cadence and at the final save. The default
uses one fixed rank-zero data batch, one seed, six generated prefix frames,
eight subsequent generated frames, and input perturbation RMS `.01`:

- perturb each of the three history frames supplied to the first target's
  scoring window (generated prefix frames 4,5,6);
- perturb the sink;
- perturb reference properties 0,4,8; configure all nine if desired.

Paired trials share generation noise and normalized perturbation direction.
History perturbations enter when that frame becomes completed history, leaving
its earlier present-slot states untouched. Sink/reference perturbations enter
before startup, including their effect on prefix generation. Output JSON gives
per-frame and per-row latent RMS differences, amplification relative to input
RMS, and nonfinite-output indicators over the continuous autoregressive rollout.
This is a finite-perturbation, finite-horizon diagnostic, not a full Jacobian or
a quality metric. It evaluates stabilizer endpoints; it does not decode video
or add resolver-quality evaluation.

`eval_weights='raw'` bounds default work; `'both'` repeats the probes under EMA.
RNG and model mode are restored afterward. Only rank zero runs the unsharded
student probe, while other ranks wait. Evaluation receipts are separate from
the model checkpoint and do not rewrite its completion marker. Retiring older
optimizer shards creates a new completion marker and a pruning receipt while
preserving the weights. A resume repairs missing evaluation receipts
before advancing the data cursor. Probe wall time and generation performance
have not been benchmarked; the horizon/seed/reference-site settings control cost.

## Verified reference bank and teacher provenance (October 18, 2026)

The intended run is **`vanshjjw/aloe-bimodal-studio-9-sinks`, run 1,
step 10424 EMA**. Its [saved training config](https://huggingface.co/vanshjjw/aloe-bimodal-studio-9-sinks/blob/976713a1eb71a619d8ed09413e63e3d28b0c4cf8/run_1/step10424/aloe-bimodal-studio-9-sinks.toml)
sets `refs.bank = '/home/vansh/aloe-studio/reference-bank-256x480-muon-encoded'`
and `refs.folder = 'bank'`. The [seen sample config](https://huggingface.co/datasets/vanshjjw/aloe-samples/blob/caf56bac76994afa33682b91eba47e93dd9acb7b/samples/aloe-bimodal-studio-9-sinks/step10424-ema-seen16/sampler-ema-seen16.yaml),
app config, and Beatles config all use that bank's `bank/` directory and
the step-10424 EMA for both modes. All 48 sample metadata rows (36 actors)
match the local bank's nine-slot presence masks, including missing slots.

**Identity limit:** these config paths and presence masks do not prove that
the local tensors are the exact bytes used by that training run. The full
pinned model and samples repositories contain no training-bank checksum
manifest, selected-reference indices or reference hashes; the EMA header
contains only `format = 'pt'`. The local checksum audit below establishes
integrity against the manifest stored with this copy. Exact historical
identity still needs an independent manifest or original training-bank copy.
Regenerating from HF sources can check reproducibility of the recipe, but
does not by itself establish which bytes the historical run consumed.

The usable copy on paramfs is:

```text
/mnt/paramfs/param/outputs/reference-compression/data/reference-bank-256x480-muon-encoded/
  bank/<uuid>/property_<1..9>/latents.safetensors
  bank/bank.csv
  bank/SHA256SUMS
  beatles-eval/...
```

The joint loader uses **`train_split = 'bank'`**, the equivalent of the original
trainer's `folder = 'bank'`. The benchmark uses this audited bank while the
deployment config names the separate rebuild output; production is gated on
that output's validation.
The archive at `/mnt/paramfs/vansh/richmond-7/reference-bank` contains the
recipe and builder code, not the encoded data. The target run's Beatles
sample YAML also uses `bank/`; its separate `beatles-eval/` selection is not
a substitute.

The historical combined studio/app copy passed full read-only validation:
1,755 actors, 159,782 images in 14,102
property files in `bank/`, plus 158 images in 35 files in `beatles-eval/`.
Fresh filesystem counts give 918/1,755 actors with at least one reference in
all nine slots (887/1,678 actors in the archived combined studio/app training
CSVs). The complete folder is 7,449,572,241 bytes: 7.450 GB / 6.938 GiB;
`bank/` alone is 7,442,213,018 bytes. The separate Beatles selection has all
nine slots for 3/4 actors. See `fresh-counts-and-size.json` for the measurement.
All 14,139 checksum entries (including both CSVs) matched. Every latent's
header, metadata identity, property, count and CSV indices matched; all
BF16 values are finite. Each file has key `video`, shape `[n,48,30,16]`,
and 1–20 images. For CSVs in `/mnt/paramfs/vansh/aloe-conversational/csvs`,
studio training coverage is 463/463 actors and selected-app coverage is
1,432/1,432. Individual properties may
be absent and use pad semantics. The actual `AloeChunkSource` loader passed
CPU checks for deterministic selection and missing-property masks.

The recipe encodes 256×480 RGB images through the normalized Wan2.2 VAE
single-image posterior-mean path. Properties 1–4 use face-centered crops;
5–9 use rounded 2×2 downsampling of the full 512×960 frame. Property 5 is
`hands_visible`, property 6 is `hands_interacting`. This is an actor/property
bank, not the older per-track `refs.safetensors` format. All 4,620 retained
old-v2 property files are byte-identical in the expanded bank; stricter
new-builder thresholds do not apply retroactively to inherited selections.
The removed actor `d5476b29…` is not in the current training CSV.

The existing copy is readable and structurally compatible; exact run identity
remains unverified as described above. For regeneration, use a separate
output directory and follow the archived `RECIPE.md`, `sinks.md` and builder
code, preserving
selection, exclusions, crops, slot order and encoder normalization. Fetch
source images and videos from HF bucket `InternalCan/aloe-conversationa-data`,
prefixes `reference-images-512x960-v1/` and `reference-videos-selected-actors/`.
The archived `fetch_gcs_node*.py` scripts need an HF acquisition replacement
before reuse. Existing `gs://` strings in `bank.csv` are historical source
provenance; training reads local encoded tensors.

Audit receipts, reproduction scripts and downloaded HF configs are under
`artifacts/reference-bank-audit-20261018/` in the audited checkout:
`expanded-bank-audit.json`, `bank-lineage.json`, `loader-smoke.json`, and
`hf-provenance/{provenance,target-lineage,sample-slot-coverage}.json`.
The HF provenance records pin repository revisions and downloaded-file hashes.
This audit did not download teacher weights or launch training. The selected
teacher config directly confirms `.8/.2` loss weights, electron-v3 poses and
local Mimi codes. It additionally specifies a 10-second audio window and
audio-history dropout settings. Its resolver was trained with `.2/.4` bands,
and the sample YAMLs start resolving at `.4`; this joint experiment specifies
a one-step resolver from `.2`, as specified for this experiment. The subsequent
local audio-conditioning review passed: joint and inference loaders preserve
Mimi's 250-token causal window per layer (10 seconds at 25 Hz), the same four
encoder layers, two-token alignment drop and four-token frame gather. Their
750-code table slices and shifted frame indices also agree. Saved random
audio/history dropout settings are training augmentations, not required
inference conditioning; joint training retains its specified dropout and
guidance behavior. The review and local source/config hashes are recorded in
`artifacts/joint-nine-ref-20261018/teacher-audio-conditioning.json`. This is a
source/config review, not a new model-output comparison. Bank validation does
not validate the full distillation implementation.

## Fresh HF-only reference-bank rebuild (October 18, 2026)

The user requested regeneration after the audit and then restricted both bank
actors and reference sources to **studio data only**. The rebuild is separate
from the existing bank and does not establish historical training-byte identity.
Shared inputs remain under
`/mnt/paramfs/param/outputs/reference-bank-rebuild-20261018`. The studio scope,
verified measurements and stricter comparison remain in `studio-only/`;
the finalized production selection and encoded output use `studio-training-policy/`.
The production bank path is
`/mnt/paramfs/param/outputs/reference-bank-rebuild-20261018/studio-training-policy/reference-bank-256x480-muon-encoded`.
Check `studio-training-policy/build-state.json` and that bank's final coverage report
before using the output.
**A running job or a populated output directory is not a completed bank.**

The scripts are in `scripts/reference_bank_rebuild/`. Production now follows the
**earlier studio training selection policy**, using fresh verified HF images and
video measurements. The archived final `RECIPE.md` and `sinks.md` describe stricter
expansion rules; the later builder retained earlier studio actors instead of
reselecting their references. Applying those later thresholds to every studio
actor changed the bank's selection policy. That strict rebuild remains available
for audit under `studio-only/` and is not the intended production bank.

The historical studio selector uses face blur ≥40 (rather than ≥60), hand
sharpness ≥60 (rather than ≥80), wider accepted face turns, and the contact gate
plus face checks for P6 without the later extra hand checks. P8/P9 do not require
a detected clothing target. `historical_studio_select.py` replays the pinned
archived studio filters and ranking; Beatles selection preserves the original
ranking and the fixed held recordings. `historical_video_select.py` supplies the
corresponding archived scanned-frame filters, crops, deduplication and append
order. Current studio membership, holdouts, source/frame identity, valid poses,
finite required inputs and encoder validation still apply. Those safeguards can
exclude candidates admitted by the permissive archived code; this replays the
historical quality and ranking policy over current verified candidates, not the
exact historical candidate pool. Original measurements, images and
scans are retained; prior encoded bank tensors are not reused.

The current studio-only inventory contains **543 actors
(463 train and 80 unseen), 8,898 source tracks and 17,274 corpus chunks**.
Four unseen actors have no eligible excess tracks;
they remain explicit zero-coverage roster entries. Quality-empty slots remain
empty. The separate Beatles bank keeps the documented historical 16 held rows
fixed and excludes their entire recordings. Final validation found **539 actors
with images and 399 with all nine slots**. Among the **458 current training
actors, 366 have all nine slots and none lack references**.

The verified upstream-image comparison selected **40,140 images** with all nine
slots for **326 actors**, versus 290 under the stricter image rules. All 36 actors
whose strict missing slots lacked an HF video regained those slots from their
studio upstream images under the historical policy. The separate historical
Beatles selection has 158 images. These are intermediate selection counts, not
the final encoded bank's coverage or size. Full-corpus comparisons against the
archived studio selector found no differences in pass flags, selected images or
indices. Evidence and exact source traces are in
`artifacts/reference-bank-audit-20261018/studio-policy-comparison.json`,
`studio-policy-notes.md` and `historical-studio-selection-checks.json`.
Policy reproduction does not establish byte-identical historical training data.

After video selection, the new actor bank has 54,490 references, versus 54,492
in the old bank restricted to these same 543 actors. Every actor's occupied-slot
mask agrees; 399 actors have all nine slots, including 366 of the 458 training
actors. Exact source identity and crop agree for 54,482 references. Seven old
upstream images are now excluded by current studio blackband labels; five have
qualifying replacements. Two video replacements are explained by historical
four-significant-digit CSV timestamps changing the 10-second deduplication
boundary. One replacement is an equal-rank ordering tie whose historical input
ordering is unavailable. Both banks' selected tracks for these actors are
entirely studio sources. The compact source comparison and bounded diagnostics
are `historical-studio-source-parity.json` and
`historical-studio-source-differences.json` in the audit artifacts. These are
source/selection checks, not latent or pixel byte-equivalence claims.

`prepare_studio_scope.py` derives `studio-only/csvs/` from exact
recording/speaker/chunk membership in the archived `all-studio.csv`, verifies
actor/session/path identity, and empties both app CSVs while preserving their
headers. All source tracks must be disjoint from `all-app.csv`; an actor-only
filter would be insufficient because 231 actors have both studio and app data.
Studio training, seen/unseen evaluation, excess and all 40 Beatles rows remain
byte-identical, preserving the historical 16 held row indices. All 14,414 current
training clips (7,586 tracks, 458 actors) are studio members with zero app-track
overlap, and every training actor is included in the bank roster. The original
CSVs, HF downloads and cached measurements remain intact. The earlier combined
inventory of 1,759 actors and 19,929 tracks is retained only as source/cache
provenance; it is not the scope of the new bank.

The audit is `studio-only/csvs/studio-scope.json`. Repository copies of that
audit, the scoped inventory and the six focused regression-test results are in
`artifacts/reference-bank-audit-20261018/studio-{scope,inventory,scope-checks}.json`.

All newly acquired inputs come from HF, with remote size/Xet hash and local
SHA256 receipts. The current named HF credential `oauth-paramluhadiya` can read
`InternalCan/aloe-conversationa-data`; using it does not change the active HF
login. Source PNGs are also checked against the manifest and selected-pose
parquet hashes. `image_keypoints` in that parquet are **already 512×960**;
only the separate `keypoints` field needs scaling from 256×480. Video poses
come from `conv-4.3k-5mins-muon-neutrino-256x480-v1/recordings/<track>/chunks/<k>/poses.safetensors`.
There are no GCP download commands in the new pipeline.

Before the scope restriction, the combined HF inventory found reference videos
for 2,116 roster tracks across 1,170 actors. Videos are required only for actors still missing a slot after
upstream image selection, so this overall coverage count is not itself a build
blocker. The exact post-selection scan targets are checked separately;
the planner ranks eligible tracks by usable chunks and continues past an
unavailable HF source until it finds up to two verified sources, matching the
archived builder. Per-actor ranked attempts, unavailable sources and selected
source identities are recorded. An actor with only one eligible track is valid;
an exhausted ranking with missing alternatives is reported separately from
selected-input failures. Two of the four previously identified missing-source
tracks remain in the studio-only inventory:
`7168b3e51724067fee4adcb52a7fbaae6ae711cb/speaker_2` and
`866721cf41e1ac227eb4bb21d0da4d9b218bd153/speaker_1`. Together they lack three
metadata files and 13 PNGs. Declared missing categories are tracked separately
from unavailable files. The original source-gap evidence is in
`hf-missing-metadata-investigation.json` and `hf-missing-png-index-comparison.json`;
the scoped comparison is copied as `studio-source-gaps.json` in the audit artifacts.
All available work has finished. These two upstream source gaps keep
`source_complete: false` and the legacy `complete: false`, while the final
marker reports `artifact_valid: true` and `available_work_complete: true`.
Selected-source and latent verification both report zero errors; unavailable
upstream candidates are distinct from failed selected inputs.

The isolated runtime is
`/home/param/reference-bank-rebuild-20261018/.venv/bin/python`, using the archived
measurement package versions and the existing PyTorch runtime. The existing
Wan2.2 VAE and MediaPipe assets were hash checked; no model download was needed.
The first 100 real tracks passed source and measurement smoke checks, producing
403 strict references spanning all nine properties. One real image passed the
encode/decode check at 39.31 dB PSNR. Small BF16 differences across batch layouts
were measured (about 0.3% relative latent L2); fresh outputs are not asserted to
be byte-identical to a different historical GPU/runtime/batch layout.

The combined source acquisition and measurement pass has finished. The scoped
`run_rebuild.py` validates and aggregates only its 8,898 tracks from the shared
measurement cache, preserving the original cache and broader outputs. With
`--selection-policy historical-studio`, it selects historical-policy upstream and
Beatles references, requests only the chosen HF video/pose sources, scans and
decodes selected frames, encodes 40 GPU shards across five nodes, and validates the resulting
bank. Historical workspace progress belongs in `studio-training-policy/build-state.json`
and `studio-training-policy/stage-logs/`; the stricter audit's progress remains in
`studio-only/`. Original acquisition/measurement receipts remain at
the shared root. Every latent group has an atomic checkpoint bound to the selection
hash, pinned VAE, encoder implementation/runtime fingerprint, and output digest.
The distributed encoder requires matching fingerprints on all nodes and uses
the existing local VAE or its verified shared copy. Video scan checkpoints bind poses, video identity,
exclusion policy, code and measurement-model hashes. If a stage fails, inspect
its log and resume; do not treat partial output as usable training data.

Video download and scanning overlap through `download_and_scan.py`. Early
passes use only complete tracks with fresh verified HF receipts; original track
policies and source identities remain unchanged. Completed chunk checkpoints
are validated and imported before the final full-plan pass. That pass requires
every selected source and expected chunk, while preserving diagnostics for
unavailable alternatives. Existing early download jobs are registered in
`prefetch-downloads.json` and joined before the full-plan downloader starts,
preventing two downloaders from publishing the same destination. Use an empty
`jobs` list when no early downloads exist. All five nodes passed a real overlap
and cache-reuse check. A 64-worker-per-node scan achieved about 97 new chunks/min
versus 65 with 32 workers on the earlier workload; this is an observed operational
comparison, not a controlled comparison using identical chunks. The current
studio preparation uses 96 CPU scan workers per node and eight encoding GPUs
on each of the five nodes (40 encoding shards). Source planning uses 64 CPU
processes over ordered track chunks, reusing the unchanged strict planner;
selected-frame decoding defaults to 64 workers. A small real-track comparison
verified exact serial/parallel plan values, dtypes, diagnostics and Parquet bytes.

```bash
/home/param/reference-bank-rebuild-20261018/.venv/bin/python \
  /home/param/bimodal/scripts/reference_bank_rebuild/prepare_studio_scope.py
/home/param/reference-bank-rebuild-20261018/.venv/bin/python -u \
  /home/param/bimodal/scripts/reference_bank_rebuild/run_rebuild.py \
  --root /mnt/paramfs/param/outputs/reference-bank-rebuild-20261018 \
  --csv-root /mnt/paramfs/param/outputs/reference-bank-rebuild-20261018/studio-only/csvs \
  --work-subdir studio-training-policy --selection-policy historical-studio \
  --expected-actors 543 --scan-workers 96 --decode-workers 64 --gpus 8 \
  --encoder-nodes 192.168.0.4 192.168.0.9 192.168.0.7 192.168.0.8 192.168.0.6
```

Run this historical controller only with both historical image and scanned-frame
selection stages integrated and validated. Existing completed scan measurements
can be reused through `--reuse-video-scan` when their source, policy and checksum
receipts match; a stricter selection result itself is not reused as the historical
video selection. Final encoding and validation passed for **54,490 actor-bank
images plus 158 Beatles images**, across **4,655 latent groups** (4,620 actor
groups and 35 Beatles groups). All groups satisfy the normalized Wan2.2 BF16
`video: [n,48,30,16]` contract. Sampled reconstruction QA passed on **54 images**,
with mean **39.9299 dB** and minimum **32.2230 dB** PSNR. The actor `bank/` folder
contains **2,559,613,682 bytes**; the full encoded-bank directory, including the
Beatles bank, manifests, receipts and provenance, contains **2,638,552,092 bytes**
in 9,504 regular files (2.639 GB / 2.457 GiB; 2,641,092,096 allocated bytes).
Exact counts and checks are in `coverage.json`,
`actor-coverage.csv`, `qa/decode-report.json` and `artifact-validation.json`
inside the production bank. Production training has launched against this
validated bank and passed the five-student-update milestone described below.

The joint training config names the finalized rebuild output. Its
`refs.require_validated_bank=true` gate requires the final artifact-validation
receipt, completed available work, and every file in the published manifest to
match its checksum before model loading. The bank's validation, selection,
encoder and manifest hashes become part of the exact-resume identity. Source
completeness is reported separately from artifact validity; missing candidates
must remain visible in the coverage report, even when every available input
has been processed successfully. The rebuild script itself does not launch training.
After finalization, `scripts/check_joint_bank.py` passed against the published
historical workspace through the actual joint dataset on CPU. It verified all
14,414 clips and 458 training actors, historical-policy provenance, studio source
membership, reference tensor shape/dtype and paired-objective samples at ranks
0 and 39. The training-actor slot histogram is 1 actor with four slots, 6 with
six, 23 with seven, 62 with eight and 366 with nine; none have zero references.
Both rank plans agree on 168,284 student updates for the full epoch. The receipt
is `studio-training-policy/proofs/joint-loader-smoke.json` under the rebuild root.
Compact repository evidence is in
`artifacts/reference-bank-audit-20261018/final-bank-validation.json` and
`artifacts/reference-bank-audit-20261018/joint-loader-smoke.json`.

## October 18 deployment and runtime controls

The teacher EMA has been downloaded from the pinned HF revision and checked
against SHA256 `003c7f491aaab2f2dc656f1ca639d84017564cdff77b78734f85df37e5455d6b`.
Assets and run records are under
`/mnt/paramfs/param/outputs/joint-nine-ref-20261018/`. The exact original
`aloe-studio-no-blackbands-high-quality.csv` has 16,148 clips; the derived CSV
excludes both crossbleed and suspected crossbleed entirely, retaining 14,414
clips and 458 actors. `scripts/prepare_joint_training.py` records both manifest
hashes and validates the actual video/pose tensor layout `[T,48,30,16]` plus
local Mimi codebooks. All 43,242 required files passed header/read checks.
Pose-v3 files were made group-readable to `param` with the user's authorization.
No older pose latent variant is substituted.

Reference K/V is computed once inside each reference context and remains
differentiable. The four-query stabilizer layout uses fixed external tensor
shapes and tensor selectors across startup, prefix/target boundaries and tail
steps. The original trimmed path remains available for numerical comparison.
`distillation/test_joint_layout.py` checks outputs and parameter/input gradients
against the original path; `probe_joint_compilation.py` exercises graph reuse.
Compiler cache limits are 64/256, with local disk caches reused between runs.
Temporal K/V packing happens inside each checkpoint so backward can reconstruct
the packed tensors from shared inputs. Checkpoint closures retain only
non-tensor arguments; saved activation tensors are released after backward.
Completed objective graphs are released before the next objective, and garbage
collection runs at shared round boundaries before freezing long-lived objects.
Metrics include start/end allocated memory, peak memory and GC time to expose
retention between updates.

`learn_dmd_joint.py --benchmark 6 --benchmark-skip 2` runs six complete joint
rounds and reports the last four. It skips the production critic-only warmup,
W&B and checkpoints, while preserving four critic updates and one student
update per measured round. `benchmark.complete.json` records topology, timings,
memory and per-rank compiler counts. Production retains the 150-round critic
warmup. All five nodes must participate in the performance benchmark.

The repaired five-node benchmark completed six joint rounds on all 40 H100s.
Measured rounds 3–6 took 8.143, 8.096, 8.320 and 8.170 seconds (8.182-second
mean). Compilation stayed at 64 graphs on every rank, with no new graphs after
round one and no graph breaks. Allocated memory returned to 39.94 GiB between
rounds; the measured peak was 61.66 GiB. Two-second telemetry samples covering
the measured window averaged 91.54% GPU utilization and 556 W per GPU across
all five nodes. Preparation, input wait and GC averaged 3.3 ms, 0.26 ms and
1.8 ms respectively. Receipts are in
`artifacts/joint-nine-ref-20261018/benchmark-03-{complete,runtime,telemetry}.json`.
This benchmark used the audited existing bank; it is not a production launch
or evidence that the fresh bank has finished.

Production job `production-01` launched on all **40 H100s across five nodes**
at 10:17 UTC from pushed commit
`46a6da6222a73e516584a37a11be94e09384fff2`. Its immutable source snapshot is under
`/mnt/paramfs/param/outputs/joint-nine-ref-20261018/code/46a6da6222a73e516584a37a11be94e09384fff2`;
the snapshot's `experiments/distillation/joint-nine-ref.toml` has SHA256
`8bb822e4727280948c4fbc93589168db35e7a030e712c4049c5ebad9d2e6cf73`.
The job manifest is
`/mnt/paramfs/param/outputs/joint-nine-ref-20261018/jobs/production-01/job.json`,
and the run directory is
`/mnt/paramfs/param/outputs/joint-nine-ref-20261018/training/run-20261018-101840-016723`.
Its saved `config.json`, `runtime.json` and `epoch-plan.json` identify production
mode (`benchmark: false`), the finalized historical studio bank, 150 critic
warmup rounds and 168,284 student updates for the full epoch. All five original
launch commands used the same configuration and disabled W&B. This was an
authoring mistake: both the saved configuration and an unconditional launcher
override disabled it. Consequently, `production-01` originally produced **no online W&B
metrics, evaluations or artifacts**; its JSON metrics and sensitivity receipts
were retained locally. Historical sensitivity JSON has since been uploaded as
small run files, as documented below. A second original omission left sensitivity reports local even when
logging was enabled. Both paths are now corrected: production records training
metrics and every checkpoint's actual sensitivity results. Compact scalars
retain individual intervention sites; full JSON report files preserve every
seed, target frame and row. Checkpoint weights and optimizer artifacts are not
uploaded.
W&B remains disabled for benchmarks. Online preflight used the real saved round
600 losses and 655 sensitivity fields, verified their remote history/summary,
and verified same-ID resume without consuming round 601. This preflight did not
perform training. The verified run is
<https://wandb.ai/canopy-labs/param-distillation/runs/0db61f9a18bf>;
see `artifacts/joint-nine-ref-20261018/wandb-online-preflight.json`.

**Current status: production-04 is running and verified on all five nodes /
40 GPUs.** At **19:50:42 UTC (12:50:42 p.m. PDT)**, it had completed round
**1327 / student 1177 / critic 5308**, including 13 updates after restoring
checkpoint 1314. Training metrics, recovered evaluation, runtime checks,
historical archive and the saved shared workspace all passed their audits.
Open the [shared project workspace](https://wandb.ai/canopy-labs/param-distillation?nw=joint9ref26)
to see both recorded history and the continuing production run. The full epoch
remains incomplete. The job is
`/mnt/paramfs/param/outputs/joint-nine-ref-20261018/jobs/production-04/job.json`,
ID `91216545b2dd46d79b0e72bef64f6643`. It launched at **19:40:40 UTC
(12:40:40 p.m. PDT)** from pushed commit
`8ca1f9005e919d46df5be1f164cea4d2750d9d15`; all 334 tracked source files in the
immutable snapshot were verified. All ranks restored at **19:46:40 UTC
(12:46:40 p.m. PDT)** to round 1314 / student 1164 / critic 5256.

The previous job, `production-03`, exited on all five nodes at 19:35:52 UTC
after saving **round 1314 / student 1164 / critic 5256**. Its save
completed before a retention identity check raised an exception, so evaluation
and its W&B publication did not run. This was not a loss, gradient, optimizer,
or checkpoint-write failure. The full epoch remains incomplete.

The recovery checkpoint is
`/mnt/paramfs/param/outputs/joint-nine-ref-20261018/training/run-20261018-101840-016723/step-001314-17e90cdcbce1`,
UUID `23be3408cd624f94889e244e9ad1565b`. All **44 completion-manifest files**
matched their recorded sizes, including all **40 optimizer/RNG shards**; their
total is **222,754,501,187 bytes**. `latest.json` points to this generation.
Its optimizer, scheduler, EMA, RNG and data cursor support exact resume with
**no completed-update replay and no second full checkpoint save**.

The concrete retention bug was inconsistent compatibility normalization. The
original `epoch-plan.json` and older checkpoints retain
`training.save_every_n_steps=200`. The cadence-change trainer correctly treated
cadence as an operational setting and omitted that key from checkpoint 1314's
training identity. Retention still ignored only `checkpoint_keep_last`, so it
rejected the otherwise identical identity with `Checkpoint identity does not
belong to this run`. The original epoch-plan bytes still match the persisted
retention policy hash. The fix shares one compatibility normalizer between
resume and retention, ignoring the same explicit runtime keys for this joint
schema while retaining all model, data, objective and schedule checks. Saved
checkpoint metadata and the original epoch plan are not rewritten.

Only the owned `.save_q.pending` from this migration was archived, preserving
its exact bytes and digest, after checkpoint durability and all five worker
exits were verified. Its record is
`RUN_DIR/recovery/metrics-1314-20261018/durable-save-recovery.json`, with the
original marker at `RUN_DIR/recovery/metrics-1314-20261018/save_q.pending`.
The record was created with status `pending_startup_completion`: archiving
the active trigger prevents an unnecessary save-and-exit at round 1315 and
does not itself prove evaluation or online publication. Those startup gates
have now passed, as recorded below. The original pending receipt remains
unchanged; a separate hash-bound completion receipt records the successful
recovery in `artifacts/joint-nine-ref-20261018/checkpoint-1314-recovery-complete.json`.
Do not recreate or generically discard other save requests.

Production-04 applied the corrected retention policy: all eight completed
weight generations remain, only checkpoint 1314 retains optimizer state, and
checkpoint 1200's **181,837,166,878 bytes** of optimizer shards were retired.
It then computed the missing sensitivity evaluation and verified all **22
compact scalars** plus the full raw report and definition JSON by remote
SHA256 before any training update. `production-04/node-0.log` line 49 records
`checkpoint_evaluation_recovered`; its verified publish token is
`e53d6ddb6f6149ddb915fdf089582162`. Startup restores the saved training RNG
after tracking/evaluation. No second full checkpoint save was required.

The first new update was **round 1315 / student 1165 / critic 5260**, with
no completed-update replay or extra save. The online audit compared all **98
canonical training fields** against the local metrics for eight consecutive
rounds, **1315–1322**: all 784 values and all 98 summary fields matched, with
no missing fields. It also reverified the 22 sensitivity summary values and
downloaded report/definition hashes; no checkpoint artifacts were uploaded.
Later rounds were outside that bounded remote-history snapshot, not declared
missing. See `production-04-online-metrics-verification.json` under
`artifacts/joint-nine-ref-20261018/`.

The local runtime audit covered rounds **1315–1327**. The first update took
71.076 seconds including compilation. Subsequent rounds **1316–1327** had a
median of **8.4911 seconds**, ranging from **8.3685 to 8.6033 seconds**.
Compilation stayed at **64 graphs, zero graph breaks and no further graphs**;
peak allocated/reserved memory was **61.6630 / 69.4922 GiB**. These measurements
include the new diagnostics and CUDA timing hooks; they do not establish zero
instrumentation overhead relative to earlier roughly 8.2-second rounds. See
`artifacts/joint-nine-ref-20261018/production-04-runtime-verification.json`.

Periodic saving remains every **1,000 total rounds**, so the next periodic save
is **round 2000 / student 1850**; final and explicit saves remain enabled.
The shared-normalizer and recovery source checks passed 37 focused CPU tests
and matched the deployed source; see
`artifacts/joint-nine-ref-20261018/production-04-recovery-source-validation.json`.
The [shared project workspace](https://wandb.ai/canopy-labs/param-distillation?nw=joint9ref26)
selects both the verified historical archive and the continuing run. Its saved
specification and remote series were API-verified; browser rendering was not
independently inspected. The production run identity remains
<https://wandb.ai/canopy-labs/param-distillation/runs/0db61f9a18bf>.

Historically, `production-03` launched from pushed commit
`993e56c32207399e898be9d501699cb69b5d36d2` and restored all 40 ranks at 19:06 UTC
to **round 1200 / student 1050 / critic 4800**. Its immutable snapshot had all
314 tracked files verified, and its local/W&B configs both used cadence 1000.
The first resumed round was 1201; round 1202 took 8.173 seconds with finite
losses, 64 graphs, no graph breaks and no new compilation. Those earlier
observations remain in `production-03-cadence-resume.json` and
`production-03-wandb-cadence-verification.json` under
`artifacts/joint-nine-ref-20261018/`; they do not describe a currently active
job or verify the new metric instrumentation.

The preceding `production-02` recovered from checkpoint 600, then completed
periodic saves at rounds 800, 1000 and 1200. Its recovery evidence remains in
`production-02-resume.json` and `production-02-wandb-verification.json`.
The cadence change reused its already active round-1200 save. After checkpoint,
evaluation and remote publication completed, all owned workers stopped with
the last completed round still 1200. No extra save or completed-update replay
was needed. The durable checkpoint used is
`/mnt/paramfs/param/outputs/joint-nine-ref-20261018/training/run-20261018-101840-016723/step-001200-a376498524eb`.

The original `production-01` stopped after round **699**, with
**549 student / 2,796 critic updates**.
At about 12:42:25 UTC, rank 5 stopped reaching collectives after its DataLoader
queue feeder reported a missing POSIX shared-memory file. The other 39 ranks
waited until the two-hour NCCL timeout, and owned-job cleanup finished at
14:44:05 UTC. The later nonfinite-gradient error occurred after collective
abort and does not establish an initial numerical failure. Evidence strongly
supports logout-driven IPC cleanup: all five hosts had `RemoveIPC=true` and
`Linger=no`, and node 0 repeatedly stopped the `param` user manager as SSH
sessions ended. The exact process that unlinked the file was not audit-traced.
See `artifacts/joint-nine-ref-20261018/production-01-failure-diagnosis.json`.

The recovery point for the failed `production-01` was
`/mnt/paramfs/param/outputs/joint-nine-ref-20261018/training/run-20261018-101840-016723/step-000600-58b63bd44fe8`:
**round 600 / 450 student / 2,400 critic updates**, after all 150 warmup rounds.
Its completion identity, all 44 file sizes, 40 optimizer/RNG shards and completed
evaluation receipt were checked. The 99 completed student updates after that
checkpoint were not saved and were replayed by `production-02`; round 699 was
not resumable. Later completed saves have retired checkpoint 600's optimizer
state under the newest-only policy; use the current `latest.json` for resume.

IPC mitigation is applied on all five nodes: `param` now has `Linger=yes` and
an active `user@1011.service`; the global `RemoveIPC` setting was unchanged.
The launcher preflight now requires both conditions. An owned POSIX shared-memory
object survived SSH logout and a 15-second quiet interval on every node, with
unchanged payload/inode and user-manager PID. Four nodes exercised the final
session logout; the fifth had other sessions open. The proof and its limits are
in `artifacts/joint-nine-ref-20261018/linger-shm-survival.json`.
The pre-resume correctness audits passed. They check the original launch and
immutable source, full filtered CSV and final bank identity, all 40 checkpoint
state shards, all 699 logged rounds, exact-resume compatibility, and bounded
rollout/loss behavior. No additional training-semantic mismatch was found. The
only effective configuration changes for that recovery were the checkpoint path, eight-save
retention, and online W&B settings. The original run configuration is preserved;
each resume writes its effective configuration, arguments, topology and source
identity under `resumes/`, linked by `latest-resume.json`. All 25 runtime,
tracking and retention tests passed; the nine retention tests also passed on
paramfs. See `production-correctness-{launch,training,data-model}-audit.json`,
`resume-compatibility-tracking-audit.json`, and `recovery-runtime-checks.json`
under `artifacts/joint-nine-ref-20261018/`. The subsequent actual distributed
restore and resumed updates passed, as recorded above. The effective resume
configuration and runtime for that recovery are in
`training/run-20261018-101840-016723/resumes/20261018-165705-662493/` under the
output root. The historical production-03 cadence-change resume is in
`training/run-20261018-101840-016723/resumes/20261018-190624-581011/`.
`latest-resume.json` identifies the newest actual resume record; the original
699 metric records and later recovery records remain. Use that newest record
and the current job receipt when assessing progress.

At **10:31 UTC**, the run had completed all 150 warmup rounds and **five student
updates / 620 critic updates**. Observed metrics were finite. Compilation held
at **64 graphs with zero graph breaks** and no new graphs after the first joint
round; allocated memory returned to **39.9435 GiB**, with a **61.6630 GiB** peak.
Joint updates 2–5 averaged **8.6586 seconds** each.
The receipt is `artifacts/joint-nine-ref-20261018/production-five-updates.json`.
Separate telemetry covering an initial 30-second production interval observed
all 40 GPUs at **87.0% mean utilization and 531 W mean power per GPU**; see
`production-initial-joint-telemetry.json` in the same artifact directory. These
production observations are separate from the earlier benchmark measurements.
The following records describe successful startup before that failure; they
do not indicate that training is currently running. The save-and-continue request
issued after six student updates completed successfully as `step-000157-7c74280771e1` in the
run directory, saving **seven student / 628 critic updates**. Its 44 manifest
file sizes were verified, including all **40 rank shards**; the checkpoint
contains **222,754,538,684 bytes**. The optimizer state was retained, sensitivity
evaluation returned `status: ok` with zero nonfinite values, the save request was
acknowledged, and training continued.
At **10:47:45 UTC**, the run had reached **10 student / 640 critic updates**.
That round took **8.132 seconds**, with 64 graphs, no new graphs or graph breaks,
and allocated memory returning to 39.9435 GiB. The evidence is
`artifacts/joint-nine-ref-20261018/production-touch-save.json`.

The first save request through completed evaluation took **953 seconds
(15 minutes 53 seconds)**. Recent periodic saves paused training for about
12–13 minutes. At round 1000, model-file creation through the durable marker
took 709 seconds, followed by 10 seconds for the local sensitivity receipt.
The 40 rank-state files contain **181.84 GB** of the **222.75 GB** checkpoint;
their serial gather/serialization/write stage took about 594 seconds. This
identifies synchronous checkpoint persistence as the dominant pause. It does
not separate network transfer, serialization, and filesystem time into distinct
benchmarks. The sensitivity calculation is a small fraction of the pause.
See `artifacts/joint-nine-ref-20261018/checkpoint-cadence-timing-audit.json`.

The current joint/full-GT trainer replaces that serial optimizer-shard save
with concurrent writes from every owning rank to the shared run directory.
Each shard is flushed and atomically renamed; all ranks exchange write status,
and rank zero verifies the complete shard set before checkpoint completion.
The rank-state format and resume reader are unchanged. Raw weight exports and
merged EMA still use rank zero. This change requires a shared output filesystem
and has not yet been timed on a full production checkpoint; the historical
timings above describe the previous implementation.

The user authorized reducing periodic saves from 200 to **1,000 total rounds**.
The deployed trainer has no live cadence polling, so the change required a
coordinated restart. Round 1200 was already being saved when the request arrived;
the migration waited for that checkpoint, its sensitivity evaluation, and remote
W&B confirmation, then reused it without requesting another save. The next
periodic save after resuming round 1200 is **round 2000 / student update 1850**.
Final/manual saves, online sensitivity publishing, eight completed checkpoint
sets and newest-only optimizer state remain in effect. Actual restore and new
updates passed, as recorded above. `checkpoint-cadence-1200-prelaunch.json`
verifies the completed checkpoint and unchanged training signature without
rewriting metadata; five cadence tests reject invalid intervals and preserve
32 incompatible training-identity guards. Periodic checkpoint 1000's actual
655 sensitivity fields also matched W&B; see
`cadence-periodic-sensitivity-verification.json` in the same artifact directory.

Touch `RUN_DIR/save` to save and continue, or `RUN_DIR/save_q` to save and exit.
Requests are claimed durably and acknowledged only after checkpoint/evaluation
completion. A newly complete checkpoint retires older optimizer states before
the sensitivity evaluation starts; the latest eight completed weight/evaluation
sets are retained. An early save-and-exit writes `training.paused.json`, not an
epoch-completion claim. At the initial retention-policy application, the run had
four completed checkpoints (157, 200, 400, 600), and only checkpoint 600 retained
optimizer state. Later saves advance the protected generation under the same
policy; use `latest.json` and `checkpoint-retention-last.json` for current state.

Once training is active, request a save and continue by touching `save` in the
active run directory. For the original run, that path is:

```bash
touch /mnt/paramfs/param/outputs/joint-nine-ref-20261018/training/run-20261018-101840-016723/save
```

To request a save and exit instead:

```bash
touch /mnt/paramfs/param/outputs/joint-nine-ref-20261018/training/run-20261018-101840-016723/save_q
```

The user authorized retaining only the latest **eight completed checkpoints**,
with optimizer state only for the newest. At the measured payload sizes, this
bounds checkpoint storage to approximately **509 GB steady state**, plus about
**223 GB transient space** while a new checkpoint is being written. The policy
was applied to the four existing checkpoints at 16:41 UTC with none pruned;
see `artifacts/joint-nine-ref-20261018/checkpoint-retention-applied.json`.
The earlier full-epoch projection in `checkpoint-storage-projection.json`
describes retaining all historical weights and is superseded by this authorized
retention policy. Local logs and other run assets are additional to these
checkpoint payload estimates.

## Handoff to the agent on the training node

1. Check out the pushed `param/bimodal` branch and read this file. Preserve the
   asymmetric student/critic prefix counts: student6+4, critic3+4. Preserve
   highest-noise-slot-only audio guidance, full batch160 per objective, common
   `.8/.2` FM/DMD coefficients, prefix-only stabilizer dropout `[0,.1,.2,.2]`,
   and the resolver's existing categorical parent dropout `[.1,.1,0]`.
2. Verify the deployment config's data roots/full CSV, actor-reference-bank root,
   nine-reference teacher checkpoint, Mimi directory, and output directory.
   The reference bank is read from `<refs.bank>/<train_split>/<uuid>/property_N/`.
   Missing properties use the existing learned-pad semantics. Do not derive
   reference images from temporal GT targets. The checkpoint loader requires
   the trained `ref_patch_embedding` and rejects unsupported extra trained
   modules rather than silently dropping them. The asset and bank are identified
   above, and the checkpoint is provisioned with a pinned hash. The selected
   run's saved config directly confirms the `.8/.2` loss coefficients. The local
   audio-conditioning review passed for the recorded configuration and source
   hashes; see `artifacts/joint-nine-ref-20261018/teacher-audio-conditioning.json`.
   Recheck those invariants if the Mimi asset, audio configuration or loader
   implementation changes.
3. Use the existing CUDA/PyTorch/FlashAttention environment and the provisioned
   Mimi frontend. Reference K/V caching is enabled; verify the benchmark receipts
   for actual full-model compilation, memory and throughput.
4. Read the current-status section above before touching any process.
   `production-03` exited after durable checkpoint 1314. Production-04 restored
   it and passed retention, missing evaluation and verified online JSON
   publication before training. Consecutive updates 1315–1327 and all 98 online
   fields for rounds 1315–1322 passed the bounded runtime/remote audits. For
   recovery use the current durable `latest.json` target;
   do not revert to 1200 or start a duplicate job. The exact owned pending
   save request is archived under `RUN_DIR/recovery/metrics-1314-20261018/`;
   do not reissue it or cause another full save at round 1315. Inspect persistent
   job status to avoid duplicate launches. Preserve original logs, online
   tracking, IPC preflight, cadence 1000 and eight-checkpoint retention.
5. For a future fresh launch, use one wrapper on each node with the same config,
   master address, port and node count. Set a distinct zero-based `NODE_RANK`
   per node. For example:

   ```bash
   NNODES=5 SLOTS=8 NODE_RANK=0 MASTER_ADDR=<master-ip> MASTER_PORT=29931 \
     ./run_dmd_joint.sh experiments/distillation/joint-nine-ref.toml
   ```

   Use a persistent job/session manager as appropriate. The wrapper does not
   SSH, source credential files, or kill other jobs. `TORCHRUN` can override its
   default `.venv/bin/torchrun`. Set cluster-specific NCCL/Gloo interfaces in
   the environment when needed. The production config now enables online W&B;
   verify authentication and the metrics/sensitivity helper before resuming.
   JSON metrics are always saved, and checkpoint artifact upload is disabled.
   Require `param` lingering and an active user manager on all five nodes.
6. Inspect `epoch-plan.json` and the first metric records for corpus size,
   derived update counts, actual global objective rows, LR, and both losses.
   Checkpoint directories are `step-<round>-<unique-id>`; `latest.json` points
   to the latest completed one. Partial directories start with `.step-` and
   have no committed generation until promoted.
7. Resume by passing `--set 'resume_from="/absolute/checkpoint/directory"'`
   on every rank. The resume reader still loads through rank zero; subsequent
   parallel saves require every rank to access the shared run directory.
   Data paths/configuration, world size, and training schedule must match.
   All optimizer, EMA, RNG and epoch-cursor state is restored; tracking
   initialization is isolated from the resumed RNG stream. A finished epoch
   performs no additional optimizer updates. Extending the epoch or changing
   topology is a separate migration, not a silent resume.

Relevant implementation files:

- `distillation/bimodal_net.py`: nine-reference embedding, RoPE and attention
  support, backward-compatible zero-reference callers;
- `distillation/joint_dmd.py`: detached/generated wavefront, four-target scoring,
  teacher guidance, resolver wrappers and stable stabilizer critic loss;
- `distillation/joint_data.py`: complete-epoch plan and reference/GT resolver data;
- `distillation/joint_eval.py`: controlled per-frame perturbation probes;
- `distillation/joint_metrics.py`: detached weighted diagnostics and the canonical
  online training projection;
- `distillation/joint_tracking.py` and `joint_sensitivity_tracking.py`: verified
  online scalars and allowlisted, digest-verified sensitivity JSON uploads;
- `distillation/checkpoint_retention.py`: shared runtime identity normalization
  and recoverable latest-eight/newest-optimizer retention;
- `learn_dmd_joint.py`: mixed optimizer steps, warmup/decay, EMA, exact resume,
  checkpoint lifecycle and evaluation cadence.

## Metric parity and the compact W&B layout

The recovered prior entrypoints are `learn_dmd_stabilizer_limits.py` and
`learn_dmd_resolver.py`, with their corresponding stabilizer/resolver helpers.
Their hash-bound inventory is in `prior-metric-parity-audit.json`. These actual
entrypoints did not automatically log videos, images, tables, histograms,
time buckets or decoded validation. Optional diagnostics in the generic
bimodal trainer are a different experiment and do not define this parity target.

The joint trainer now computes the prior DMD normalizer and teacher–critic
absolute difference for each objective; teacher–critic velocity cosine/RMS;
generated output RMS; next-step learning rates; the fraction of resolver
history attention slots dropped; numeric phase indicators; and detailed CUDA
context, input-generation, forward, scoring and backward timings. Diagnostics
use detached tensors and the same valid-row weights as the losses. Global
weighted squared sums are reduced before taking RMS. Padding contributes
neither numerator nor denominator. Timing-only hooks add no model forwards or
RNG draws; CPU parity checks cover outputs, all gradients, model-call counts
and RNG state for both stabilizer prefix lengths and the resolver. The separate
production-04 runtime audit verified the instrumented full model at 64 graphs
with no graph breaks or further compilation after its first resumed update;
measured steady timing is reported above.

The local `metrics.jsonl` keeps the existing names plus exact compatible legacy
loss/frame aliases. A normal joint update has 123 local numeric fields, projected
to 98 canonical online scalars across 21 top-level groups. The nine overview
keys are `train/round`, `train/student_loss`, `train/critic_loss`,
`train/student_updates`, `train/critic_updates`, `train/epoch_fraction`,
`train/student_lr_used`, `train/critic_lr_used`, and `train/seconds_per_round`.
Objective, diagnostic, timing, memory, gradient,
learning-rate, progress and throughput details have separate groups. The
projection rejects unknown numeric fields and conflicting aliases so future
metrics cannot silently disappear. Unqualified teacher–critic
metrics from separate older runs become explicit per-objective metrics here.
The old critic pool's multiplied batch count is not reused as a count of unique
rows: this trainer reuses one paired pool for four critic updates. Missing
historical diagnostics are omitted, never synthesized as zeros.

Sensitivity overview charts retain the individual three history sites, sink,
and three configured reference sites. Each site has maximum and final-frame
amplification, grouped under `sensitivity_history`, `sensitivity_sink`, and
`sensitivity_reference`. For multiple seeds, the maximum is the maximum of
per-seed maxima and final-frame amplification is the arithmetic mean over
measured seeds. Skipped cases are counted separately; an invalid measured seed
prevents a partial finite-only site summary. `sensitivity_overview` contains
checkpoint and health metadata. The current raw-only configuration therefore
has 14 scientific overview scalars plus eight metadata/health scalars, instead
of flattening all per-frame arrays into hundreds of default charts.

Every complete raw/EMA sensitivity JSON is preserved online as an allowlisted
small run file, with its content digest and checkpoint provenance. These files
retain all trajectories, per-row values, configuration, seed and intervention
metadata. No model, teacher, optimizer or checkpoint directory is uploaded.
The run files live under `sensitivity_reports/<original-evaluation-UUID>/`.
A small `sensitivity-overview-definition.json` beside each raw report records
the scalar mapping, site/seed support and source SHA256. Remote file size and
downloaded SHA256 must match before publication is accepted. Invalid/skipped
site amplitudes are omitted, and only their stale compact latest-summary
values are cleared; historical rows remain intact.

All seven existing completed evaluations at rounds **157, 200, 400, 600, 800,
1000 and 1200** are now inspectable in the
[run files](https://wandb.ai/canopy-labs/param-distillation/runs/0db61f9a18bf/files).
Their seven original JSON files and seven definitions total **258,990 bytes**;
all 14 were downloaded and SHA256-verified after upload. The backfill used only
public run-file APIs, with no second SDK run, scalar-history changes or weight
uploads. Pruned checkpoint IDs were accepted only through validated optimizer
prune lineage. See `sensitivity-historical-json-backfill.json`,
`sensitivity-implementation-audit.json`, and the real installed-API preflight
`sensitivity-json-upload-preflight.json` under
`artifacts/joint-nine-ref-20261018/`. Checkpoint 1314's recovered evaluation
and two verified JSON uploads were completed separately at production-04
startup; they are not included in those 14 historical files or byte totals.

Existing production history is unchanged. A separate
[canonical history archive](https://wandb.ai/canopy-labs/param-distillation/runs/archive1to1314-b3068e2fb71f)
now contains **1,314 recorded rounds and 84,950 verified scalar values**,
including 150 critic-only warmup rounds and 1,164 joint rounds. Its provenance
selects the retained attempt for each round and excludes the 99 superseded
production-01 records at rounds 601–699. Missing historical diagnostics are
omitted. The archive also contains the 22 compact sensitivity fields at all
eight completed evaluation checkpoints, derived from the validated original
reports. Remote values match the prepared payload exactly. Ingestion times
reflect archive upload time; `train/round` is the training axis. See
`historical-backfill-1314-manifest.json` and
`historical-backfill-1314-applied.json` under
`artifacts/joint-nine-ref-20261018/`.

The published [shared project workspace](https://wandb.ai/canopy-labs/param-distillation?nw=joint9ref26)
selects that archive and the continuing production run. Use this project link
to display both runs; a run-specific page restricts the view to that run. The
workspace has **25 sections and 55 charts**, covers all 98 training fields
with deduplicated metric charts and nine core overview fields, and puts the 14 scientific
sensitivity values in six comparison charts alongside separate health metadata.
Automatic panel creation is disabled. Every chart key has recorded remote data
in at least one selected run, including all eight checkpoint sensitivity series.
The saved view specification was read back exactly and its series were
API-verified; **browser rendering was not independently verified**. See
`wandb-curated-workspace-applied.json` and
`wandb-curated-workspace-verification.json` in the same artifact directory.
The full sensitivity JSON reports remain inspectable in the production run's
files, separate from these compact charts. No model or code artifacts were
uploaded.

The original visibility failure had two distinct causes. The saved project
workspace selected three older runs and excluded this run, and several training
sections were collapsed or carried restrictive chart settings. Separately, the
joint entrypoint omitted the prior diagnostics listed above. The transport audit
found all 63 then-produced scalars in remote history and summary, with ordinary
streaming lag. Changing `commit=False` is not the fix: it allows sensitivity to
join the same checkpoint round without dropping training metrics.
