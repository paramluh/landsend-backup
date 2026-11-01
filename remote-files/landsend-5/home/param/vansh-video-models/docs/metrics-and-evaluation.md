# Training metrics and evaluation

Use a small, fixed W&B metric contract for the next stabilizer and resolver run. Training produces scalar observations; evaluation produces a few summary scalars and matplotlib figures. Names, formulas, axes, and evaluation inputs remain stable across runs so comparisons retain their meaning.

Implemented for the full-GT trainer in `training_metrics.py`, `training_tracking.py`, `endpoint_sensitivity.py`, `rollout_evaluation.py`, and `training_evaluation.py`. The historical joint trainer retains its original contract. CPU checks cover numerical reductions, paired sensitivity, media encoding, checkpoint/gallery retries, and W&B payloads. Real model evaluation and live upload/playback remain runtime validation steps. Cloudflare deployment is held until the user supplies the hosting VM details.

## Runtime dependencies and preflight

The training environment must also install the pinned evaluation dependencies
from `requirements.txt`: `kornia==0.8.3`, `kornia-rs==0.2.0`, and `timm==1.0.30`.
AutoBench background scoring uses Kornia and the BiRefNet model uses timm; they
are required even though training and throughput benchmarks do not import them.
The environment path is identical on the five hosts, but its contents are local
to each host, so verify these imports on every host. Video export also requires
PyAV and the `ffmpeg` executable.

Before training, verify online access to the configured W&B project and ensure
`ZhengPeng7/BiRefNet-portrait` revision
`b6561965a70070d9143fd9e558f6ca3c481510db` is cached on the evaluation host.
Loading it with `AutoModelForImageSegmentation.from_pretrained(...,
trust_remote_code=True, local_files_only=True)` on CPU checks the pinned model
and its Python dependencies without using training GPUs. The October 20
preflight passed this CPU load and imports on all five hosts; it does not prove
the full evaluation's GPU memory peak or W&B video playback.

## W&B logging contract

Publish only the explicitly listed keys. Remove progress, phase, batch, timing, memory, compilation, gradient, dropout, and diagnostic alias groups from the default dashboard. Do not expand arbitrary dictionaries into W&B metrics, use automatic gradient watching, or enable system statistics as another stream of dashboard metrics. Internal resume bookkeeping and measurement counts stay in local records; fixed configuration and provenance belong in run metadata.

Use `train/step` as the single training x-axis: one completed outer training round, including critic warmup. It is an axis, not another dashboard panel. Publish each training scalar at most once per round. A round currently contains four critic optimizer updates and one student update after warmup. Missing student observations during warmup stay absent; never insert zero or repeat the last observation.

Evaluation uses `eval/checkpoint_step`, the training step of the weights actually evaluated. This must remain correct if evaluation finishes after training advances. W&B's internal logging step stays monotonic; associate eval values with their custom checkpoint axis rather than trying to write backwards into W&B history. Save a checkpoint identity with every local evaluation artifact.

Register an explicit metric schema version and evaluation protocol version in run configuration. Freeze bucket edges, units, reduction rules, perturbation settings, raw/EMA choice, and the definitions below. Formula changes require a new version and explicit comparison handling. Keep the same dashboard template and panel order across runs. Do not silently merge incompatible historical metrics into the new series.

## Training scalars

The default contract contains **54 scalar series**, plus the step axis. Counts and per-frame values do not create additional W&B series.

### Overall losses and learning rates

| Key | Meaning |
|---|---|
| `train/dmd_loss` | Weighted total student DMD objective |
| `train/dmd_stabilizer_loss` | Unweighted stabilizer DMD mean |
| `train/dmd_resolver_loss` | Unweighted resolver DMD mean |
| `train/critic_fm_loss` | Weighted total critic training objective |
| `train/critic_fm_stabilizer_loss` | Unweighted stabilizer endpoint FM mean |
| `train/critic_fm_resolver_loss` | Unweighted resolver endpoint FM mean |
| `train/student_lr` | Learning rate actually used for the student update |
| `train/critic_lr` | Mean learning rate actually used across the round's critic updates |

For this full-GT run, both totals use the selected equal average: `0.5 * L_stabilizer + 0.5 * L_resolver`. Apply these coefficients once after per-objective normalization; do not divide by two again. There is no interval-width or gradient-based reweighting. Keep the intrinsic endpoint factors and DMD normalizers unchanged. The upcoming full-GT trainer optimizes endpoint FM for both critic objectives. The older joint trainer's resolver used velocity FM; its total is not interchangeable with this one. Record the explicit coefficients in run configuration so comparisons retain their meaning.

### Timewise buckets

Bucket the **actual sampled scoring/noising time**, not the target's frame index or student rollout step. Every bucket value averages valid observations in that bucket during the current round.

| Objective | Fixed bucket identifiers | Intervals |
|---|---|---|
| Stabilizer | `t_0.20_0.40`, `t_0.40_0.60`, `t_0.60_0.80`, `t_0.80_1.00` | `(0.20,0.40]`, `(0.40,0.60]`, `(0.60,0.80]`, `(0.80,1.00]` |
| Resolver | `t_0.00_0.05`, `t_0.05_0.10`, `t_0.10_0.15`, `t_0.15_0.20` | `[0.00,0.05]`, `(0.05,0.10]`, `(0.10,0.15]`, `(0.15,0.20]` |

The current stabilizer sampler excludes tiny endpoint guards. The resolver currently samples from `0.02` to `0.20`, so its first bucket only observes `0.02` through `0.05`. Logging does not change that sampling distribution or invent observations below `0.02`.

For each objective and bucket, publish:

| Key pattern | Definition |
|---|---|
| `dmd_<objective>/timewise_loss/<bucket>` | Mean DMD surrogate loss |
| `dmd_<objective>/timewise_normalizer/<bucket>` | Mean effective denominator actually used by DMD |
| `critic_fm_<objective>/timewise_endpoint_loss/<bucket>` | Mean endpoint FM loss |
| `critic_fm_<objective>/timewise_velocity_loss/<bucket>` | Mean raw velocity FM diagnostic |

Here `<objective>` is `stabilizer` or `resolver`. These are 16 DMD and 16 critic FM series. There are no framewise variants.

For one target, let `u_t` be the analytic corruption-path velocity and let `v_T` and `v_C` be teacher and critic velocities on the same noisy input. The DMD normalizer and surrogate are:

```text
n = mean_CHW(abs(v_T - u_t))
d = max(n, 1e-8)
g = (v_T - v_C) / d
L_dmd = 0.5 * mean_CHW((y - stopgrad(y - g))^2)
```

Log the effective `d` under `timewise_normalizer`. Keep raw denominator sums, effective sums, floor-hit counts, and bucket counts in the local measurement artifact. The DMD loss measures the surrogate's magnitude; it is not a likelihood or direct video-quality score.

For endpoint `h` (`0.2` for stabilizer, `0` for resolver), define:

```text
a = (1 - t) / (1 - h)
s² = t² - h² * a²
k_h(t) = (1 - h) * s² / t
endpoint_loss = mean_CHW((k_h(t) * (v_C - u_t))²)
velocity_loss = mean_CHW((v_C - u_t)²)
```

Preserve the implementation's algebraically stable endpoint residual. Calculate velocity loss separately as a detached diagnostic; do not recover it by dividing averaged endpoint losses by an averaged scale. Large raw stabilizer velocity loss near `t=0.2` can reflect the corruption path's large analytic velocity, so read it alongside endpoint loss and its time bucket.

### Velocity probes

Publish seven scalars under each of `velocity_probes/stabilizer/` and `velocity_probes/resolver/`, without time or frame expansion:

| Suffix | Definition and purpose |
|---|---|
| `student_rollout_rms` | `sqrt(E[v_S²])` over the target velocities already evaluated by the student sampler; eight stabilizer evaluations or one resolver evaluation per target |
| `teacher_rms` | `sqrt(E[v_T²])`; teacher velocity scale |
| `critic_rms` | `sqrt(E[v_C²])`; critic velocity scale |
| `critic_minus_teacher_mean` | `E[delta]`, where `delta = v_C - v_T`; signed overall bias |
| `critic_minus_teacher_variance` | `E[delta²] - E[delta]²`; residual spread |
| `critic_minus_teacher_channel_bias_rms` | RMS of the 48 channel means of `delta`; detects channel biases that cancel in the overall mean |
| `teacher_critic_cosine` | Mean per-target cosine between flattened teacher and critic velocity vectors; directional agreement |

Diagnostics must not trigger additional model forwards. Teacher and critic statistics reuse their existing scores on identical noisy input, time, context, pose, audio, and reference conditioning. Student RMS reuses detached target velocities from the existing student rollout, with weighted element sums and counts across its solver evaluations. It describes the sampler's velocity scale; it is not a same-input comparison with teacher or critic scoring velocities. Exclude detached critic-pool generation from the student rollout statistic. Preserve training RNG and do not retain diagnostic computation graphs.

This contract is `full-gt-training-v2`. The earlier v1 `student_rms` used an extra same-input student scoring pass. Retire that key rather than silently changing its meaning; a resumed run crossing the schema boundary receives a linked W&B run with the new schema.

These statistics indicate scale, alignment, and bias. A small mean or large variance alone cannot establish that the difference is independent random noise or truly unbiased.

### Reduction and missing observations

Accumulate weighted sums and counts across microbatches, critic updates, and ranks, excluding padding; divide once globally. Compute RMS, variance, and channel-bias statistics after the appropriate global reductions. Use FP32 or better accumulation; handle only numerical roundoff when a computed variance is slightly negative. Do not average rank averages when counts differ.

All ranks allocate the same ordered bucket accumulators even if a rank sees no examples in a bucket. A globally empty bucket has no observation and must appear as a gap, never zero or a forward-filled value. Keep counts locally so plots can be audited without adding count panels. Handle zero-norm cosine inputs explicitly and record their valid count locally.

## Evaluation sensitivity plots

Each evaluation round produces one stabilizer figure and one resolver figure per configured weight variant. Use fixed W&B image keys:

```text
eval/sensitivity/<weights>/stabilizer_context_gain
eval/sensitivity/<weights>/resolver_context_gain
```

`<weights>` is `raw` or `ema`; retain the run's configured choice across evaluations. Use a panel for each denoised target slot: the stabilizer's four present slots, and the resolver target in its ordinary two-frame-history query. If several resolver targets are evaluated together, give each its own panel instead of collapsing their results.

Within each panel, y is perturbation gain. Temporal x categories follow the inputs the model actually receives, newest to oldest:

| Objective | Temporal categories | Separate final category |
|---|---|---|
| Stabilizer | `previous 1`, `previous 2`, `previous 3`, `sink` | `reference bank mean` |
| Resolver | `previous 1`, `previous 2`, `sink` | `reference bank mean` |

The reference bank has no temporal ordering: separate its category visually from the history axis. Label predecessor identities relative to the target, even when a predecessor occupies an earlier present slot. No fourth predecessor or model context expansion is requested.

### Gain measurement

Use a fixed evaluation panel of source clips, target positions, seeds, and perturbation directions. Measure the target's **completed latent endpoint sensitivity**, with the stabilizer ending at `t=0.2` and resolver at `t=0`. Freeze each panel's starting noising time, denoising schedule, and baseline context tensors in the protocol. Starting from a saved baseline query state, reintegrate only the target with and without one conditioning intervention. Replay every other context trajectory unchanged; keep pose, audio, validity, and target noise fixed.

For a temporal input identity `j`, define its perturbed value at every remaining target query `s` as `context_perturbed_j(s) = context_baseline_j(s) +/- delta_j`, using the same fixed latent perturbation while that identity is visible. Static sink/reference interventions similarly retain their fixed perturbation throughout the target's remaining denoising. This defines endpoint sensitivity with replayed context; it does not include additional AR propagation through regenerated conditioning frames.

For input site `j` and target slot `q`, use:

```text
gain(q, j) = RMS(endpoint_perturbed[q] - endpoint_baseline[q])
             / RMS(actual_injected_latent_perturbation[j])
```

Use latent perturbation RMS `0.01`. Average fixed positive and negative perturbation-direction trials. Set `sensitivity_validate_half_rms=true` for the initial real-model amplitude check; ordinary checkpoint evaluations leave it false. CPU checks cover the half-amplitude calculation, but do not establish linearity for the actual model. Rebuild affected history/reference caches and any representations that depend on the changed sink or reference; changing an input without changing its cached representation is not a valid intervention.

For `reference bank mean`, intervene on **each valid supplied reference separately**, compute its gain, then average within that sample before averaging across samples. This includes every reference the model receives, rather than the current subset of three properties. It does not mean perturbing all references simultaneously or traversing every stored candidate image in the actor's bank. Missing or invalid references are excluded, never assigned zero gain. Fixed reference draws and their hashes belong in the evaluation protocol.

Plot mean gains with spread across the fixed clips/seeds; include sample counts and intervention settings in captions or the saved data. Keep category ordering, units, and panel layout fixed across checkpoints. Save detailed measurements as JSON/NPZ alongside the figure, without creating individual-reference, individual-frame, max/final, or per-site sensitivity scalar series in W&B.

The existing sensitivity evaluator measures propagation into later AR outputs, with different intervention start times for history versus sink/reference probes. Its results cannot simply be relabelled as this context-sensitivity protocol. Keep the new protocol version explicit.

## Long rollout evaluation

At every evaluation round, generate approximately **30 seconds** with the evaluated student's actual stabilizer and resolver inference path. Start with one fixed, eligible 30-second evaluation clip; any larger fixed panel is an explicit evaluation configuration change. Use generated history after initialization, without GT feedback. Freeze the source interval, seed, bank selection, audio/pose inputs, sampler schedule, resolution, and raw/EMA choice for cross-run comparison.

At 25 fps, 189 Wan latent frames including the sink decode to `4 * 189 - 3 = 753` video frames, approximately 30.12 seconds. Require a matching source interval with sufficient GT and conditioning; do not loop, pad, or silently substitute a shorter clip. Store actual frame timestamps. Exclude the supplied sink/startup frames from summaries of generated continuation.

The current long renderer uses a frozen teacher resolver with ten steps. Replace that path for student evaluation; otherwise the video would not measure the resolver being trained.

### AutoBench background drift

Run AutoBench on full-quality generation before website transcoding, sampled at 1 Hz. Retain its current M1 and M2 definitions:

- **M1 background colour drift:** CIE76 colour difference between the frame and that video's own sink over their common background.
- **M2 background texture drift:** signed log ratio of fine-detail energy relative to the sink. Zero means unchanged; positive is noisier and negative is smoother.

Save the per-second curves, with aligned GT baseline curves, in one matplotlib figure with separate M1 and M2 panels under `eval/drift/<weights>/background_curves`.

Expose only four summary scalars under `eval/drift/<weights>/`:

```text
background_color_mean
background_color_tail
background_texture_mean
background_texture_tail
```

`mean` covers valid generated-continuation seconds; `tail` covers the final five seconds of the actual rollout. Average within each clip before averaging clips. Preserve valid masks and counts locally; missing background measurements remain missing. Texture summaries remain signed and should be interpreted against zero and the GT baseline.

Version these 30-second summaries explicitly. The existing five-minute summaries and hard-coded end window at seconds 293–297 do not apply. Retired person/identity metrics and the separate long-latency artifact detector do not enter the default dashboard.

### Framewise divergence from ground truth

At every decoded frame, compare generated RGB with GT on the exact same timestamp, crop, and resolution, with RGB scaled to `[0,1]`:

```text
L2(t) = sqrt(sum_CHW((generated_RGB(t) - GT_RGB(t))²))
```

Save a matplotlib figure under `eval/rollout/<weights>/gt_l2_curve`, with elapsed seconds on x and RGB L2 on y. Use fixed 256 by 480 images so the norm is comparable between runs. This is Euclidean L2; do not silently substitute MSE or RMSE. If a normalized version is later preferred, name and version it explicitly.

Prefer aligned original camera GT. If only Wan-decoded GT latents are available, label the curve and website as comparison with **VAE-reconstructed GT**, and retain that definition across compared runs. Compute from decoded RGB before lossy web encoding. Keep the sink visible on the curve but identify it as supplied context. Pixel L2 measures aligned divergence and is sensitive to motion/phase differences; interpret it alongside playback and background drift.

## Figures and the website

Render matplotlib figures to PNG and log them as `wandb.Image` under the same fixed key every evaluation, with checkpoint step and weight identity in the caption. Preserve figures and underlying arrays in immutable checkpoint-specific directories. Use the W&B media/table step history to browse evaluations; if using a table stepper, bind it to the saved checkpoint step. Do not add checkpoint numbers to metric names. Close figures after saving to avoid accumulating plotting state.

Each training run gets a distinct website URL and storage prefix, based on its unique run identity. Within it, an evaluation/checkpoint selector exposes:

- Full-duration generation with audio and synchronized GT/generated comparison.
- The sensitivity, background-drift, and framewise L2 figures.
- Raw/EMA labels, checkpoint identity, source clip, seed, and reference-bank identity.
- Downloadable measurements and full-quality originals, plus the W&B run link.

Preserve prior evaluation rounds when publishing another one. Use the repository's existing manifest-driven static gallery pattern. Publish only complete evaluation entries; verify uploaded media hashes and browser playback before making an entry selectable. Website transcodes are playback copies, while metrics always use the original full-quality output. The run's website URL belongs in W&B metadata, not a new scalar metric.

The implemented local gallery is `RUN_DIR/gallery/`, backed by retained copies of
the immutable files in `RUN_DIR/evaluations/`. Checkpoint directories remain flat
and can be retired without deleting evaluation history. Generation and GT have
synchronized controls; GT playback is muted to avoid duplicate audio. Cloudflare
hosting and the VM upload path will be wired after the user provides the VM.

The checked-in evaluation uses one source clip, seed 1729, 189 latent frames,
25 fps, 256-pixel width × 480-pixel height RGB, and the pinned Wan2.2 VAE SHA256.
The evaluator checks that the 30×16 spatial latents decode to this portrait
geometry without resizing or transposing the source. A run-level
`evaluation-protocol.json` fixes source, configuration, and tensor fingerprints.
The sensitivity start times are `.4`, `.6`, `.8`, and `1` for the four stabilizer
panels and `.2` for the resolver. The six startup frames are generated, and each
intervention reintegrates only its target through the remaining Euler steps.

## Implementation and acceptance

1. Replace the existing W&B projection with the allowlist above; define the schema and dashboard once. Disable legacy sensitivity scalar projection and unrelated automatic metric streams.
2. Add weighted time accumulators at loss production and detached observations of existing rollout/scoring velocities. Diagnostics must add no model forwards and must leave loss computation, gradients, and training randomness unchanged.
3. Implement the paired endpoint sensitivity protocol and fixed matplotlib layout using the actual three-frame/two-frame contexts.
4. Add the 30-second student rollout, timestamp-aligned GT curve, and duration-aware AutoBench summaries.
5. Save each evaluation atomically, log its fixed image/scalar keys, and append its complete entry to the run's website.

Acceptance checks cover exact metric key sets; no unexpected namespaces; bucket boundaries; unequal-rank counts and padding; globally empty buckets; warmup gaps; correct applied learning rates; and unchanged model-forward counts, outputs, training RNG, and gradients when diagnostic observation is enabled. Sensitivity checks cover actual injected RMS, cache invalidation, all-valid-reference averaging, paired inputs, and a known zero-response control. Eval checks cover student resolver selection, frame/time alignment, absence of five-minute summary assumptions, fixed plot keys across checkpoints, and preservation of earlier website entries. No historical W&B runs are rewritten by this change.

## Implementation references

- [Training accumulators](/home/param/bimodal/distillation/training_metrics.py), [W&B contract](/home/param/bimodal/distillation/training_tracking.py), [training lifecycle](/home/param/bimodal/learn_dmd_joint.py), and [full-GT context construction](/home/param/bimodal/distillation/full_gt_dmd.py).
- [Endpoint sensitivity](/home/param/bimodal/distillation/endpoint_sensitivity.py) and [checkpoint evaluation](/home/param/bimodal/distillation/training_evaluation.py).
- [Student rollout and gallery](/home/param/bimodal/distillation/rollout_evaluation.py), [AutoBench definitions](/home/param/bimodal/docs/corruption/autobench.md), and [scorer](/home/param/bimodal/eval/autobench.py).
- [Existing gallery pattern](/home/param/bimodal/experiments/renoise_ablation_20261015/site/README.md).
- W&B documents [metric and media logging](https://docs.wandb.ai/guides/track/log/), [custom step axes and matplotlib logging](https://docs.wandb.ai/ref/python/experiments/run/), and [table history with a step slider](https://docs.wandb.ai/guides/tables/visualize-tables/).
