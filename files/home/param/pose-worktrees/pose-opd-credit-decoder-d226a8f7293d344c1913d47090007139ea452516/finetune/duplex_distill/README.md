# Pose duplex distillation

Train the production pose-duplex model using two frozen teachers on its original Aloe data. The pose monoplex teaches pose; the audio duplex teaches speech and text. All student parameters remain trainable. The approximately 1,000-word [training audit](TRAINING_AUDIT.md) describes the actual forward, gradient, and optimizer paths.

This package optimizes teacher KL. Dataset cross-entropy is logged only; a supervised finetune requires its own training recipe.

The current recipe uses **seven nodes / 56 H100s**, two complete epochs, no evaluations, and checkpoints every two hours plus the final update. **Microbatch two is selected**: global batch 112, 284 optimizer updates, and nine warmup updates. Earlier bounded profiles and the stopped 21-step run are historical measurements, not resumable training progress.

| Role | Production checkpoint | Local path |
| --- | --- | --- |
| Student initialization | `InternalCan/ppm-aloe-poseduplex-v5-final`, export 498 | `/home/param/models/imported-weights/ppm-aloe-poseduplex-v5-final/export-498-dcp` |
| Frozen pose teacher | `InternalCan/ppm-aloe-v2pure-main-17850-final`, export 200 | `/home/param/models/imported-weights/transcript-pose/xabi-original/export-200` |
| Frozen speech/text teacher | `InternalCan/ppm-aloe-samsep-vox17127-final`, export 249 | `/home/param/models/imported-weights/ppm-aloe-samsep-vox17127-final/export-249-dcp` |

The historical `xabi-original` model directory does not identify the dataset. Training uses only `/home/param/datasets/aloe-duplex`, staged and verified on all seven nodes.

## Objective and conditioning

Every supervised output uses the complete teacher distribution at temperature one:

```text
KL(teacher || student) = sum_v p_teacher(v) * (log p_teacher(v) - log p_student(v))
speech_KL = 0.25 * speech_cb0_KL + 0.75 * mean(speech_cb1..31_KL)
pose_KL   = 0.25 * pose_cb0_KL   + 0.75 * mean(pose_cb1..31_KL)
loss = (text_KL + speech_KL + pose_KL) / 3
```

Each mean uses globally summed valid-position counts for that output. Missing pose is masked without redistributing its modality weight. Text KL uses lexical weight one, equivalent to full-vocabulary KL. Detached dataset-CE diagnostics retain the source text weighting: category CE plus five times conditional lexical CE. No dataset CE contributes gradients.

The models use dataset histories and codebook prefixes, with no generated rollout. One production loader call supplies each row’s shared supported B prefix, history, positions, and conditioning decision. Student and audio teacher share the real A waveform and one frozen Voxtral encoding per row. The audio teacher excludes pose. The monoplex teacher receives its native empty A lane: PAD text, absent Mimi codes, zero presence, and no Voxtral input.

B voice conditioning has a deterministic 0.30 draw (`v2_30`); a drawn clip can be unavailable. Available clips contain 38–125 frames. The same decision persists across epochs. There is no persona conditioning or attention dropout. Missing B pose preserves speech/text supervision; dummy row slots have zero loss and count in every stream.

## Data and update schedule

`recipe.py` reads the installed export-498 configuration and verifies production identity. `production/` contains the pinned loader and model source. Provenance retains original settings alongside explicit user overrides for loss balance, shared depth sampling, and checkpoint policy.

- Each row is one conversation’s first at most **7,500 frames**, without random cropping. Short conversations remain short. Prefix and delay extend the actual model timeline beyond conversation length.
- Each epoch covers **15,898 training orientations**; two epochs cover **31,796 real examples**. The 316 held-out orientations remain excluded from training and are **never evaluated**.
- Training hosts are `landsend-0,1,2,3,4,5,12`, eight GPUs each. Gradient accumulation is one.
- CB0 uses all valid frames. Both higher-codebook decoders use the **same deterministic 10% conversation-frame sample**, aligned by speech/pose delays (both two). Missing pose has no selected pose frames. The sample repeats across epochs.
- Effective `data` and `posedup` pose sampling is 0.1; `source_loader_data` and `source_loader_posedup` preserve the source’s 1.0. The adapter changes only training pose depth indices and their counts after source row construction. Inputs, histories, prefixes, CB0 masks, and source loader remain unchanged.

| Per-GPU microbatch | Global examples/update | Updates/epoch | Total updates | Warmup updates | Dummy slots over both epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 56 | 284 | 568 | 19 | 12 |
| 2 | 112 | 142 | 284 | 9 | 12 |

Microbatch two packs two independently prepared rows into a single forward, with distinct attention segments, reset positions, prefixes, and masks. Waveforms are encoded separately. All five loss denominators include both rows across all 56 ranks. Dummy depth selections are removed before packing. Both settings preserve complete dataset coverage and the same per-rank row order; they change optimizer update count and schedule.

AdamW retains betas `(0.9,0.95)`, epsilon `1e-8`, zero weight decay, source parameter-group learning rates, trunk clipping 0.5, and other clipping 1.0. Speech/pose decoder peak LRs are `1.5e-5`/`2e-5`. Warmup is followed by linear decay toward the configured horizon. Update index zero uses LR zero while updating Adam moments.

Depth chunks contain at most 470 selected frames. Their gradients accumulate into detached tower/link roots, followed by one main backbone backward. Each student decoder layer checkpoints activations. Teachers remain frozen. Main weights/optimizer state are FP32, model computation BF16; detached roots accumulate in activation dtype. FA3 and fused optimizer/kernel paths remain enabled. Depth decoders run eagerly.

Text-head chunk size and direct-addmm implementation are canonical training flags, recorded in `config.json`. Seven-node batch comparisons use `--text-chunk-size 512 --text-direct-addmm`, previously measured in complete 48-GPU scratch profiles. The default without these flags remains chunk 32.

## Metrics

| Namespace | Meaning |
| --- | --- |
| `train/*` | Balanced teacher KL, modality losses, speech/pose higher-codebook means |
| `text_kl/*` | Full text KL and dataset-category subsets; conditional lexical KL on word positions |
| `audio_codebooks/*`, `pose_codebooks/*` | Unscaled teacher KL for cb0 through cb31 |
| `data_loss/*` | Detached dataset CE with the same balance and source weighted text CE |
| `data_text_ce/*` | Unweighted full dataset text CE, category subsets, and conditional lexical CE |
| `data_audio_codebooks/*`, `data_pose_codebooks/*` | Detached dataset CE for cb0 through cb31 |

There are 152 loss metrics. Individual codebooks appear only in their codebook groups. `text_kl/full` equals `train/text_loss`; `data_text_ce/full` differs from source-weighted `data_loss/text_loss`. No evaluation metrics are produced. The run directory contains resolved configuration, plan, initialization provenance, per-rank events/progress, and metrics. The default W&B project is `param-pose-llm`; `wandb.json` records the actual run URL.

## Launch and operate

Run from `/home/param/pose-repo` on landsend-0 using the tracked [training environment](../transcript_pose/environment/README.md) plus `runtime-requirements.txt`. Use a fresh absolute run directory under `/home/param/runs/pose-runs`; place bounded checks and profiles under `/home/param/runs/smokes`. The launcher synchronizes this package, `finetune/storage_paths.py`, and the environment scripts. It checks model/data readiness, pinned FFmpeg 7.0.2, InfiniBand, and eight free GPUs per target, then detaches the job. It does not evict other jobs.

Logs, plans, metrics, and control files belong to the run directory. Imported initialization weights and teachers live under `/home/param/models/imported-weights`. Checkpoints created by a named run live under `/home/param/models/trained-weights/<run-name>/checkpoints`; the run directory contains a compatibility link to that directory. Classified runs under `/home/param/runs/failed` retain the same model namespace.

```bash
# Synchronize and check all seven nodes without starting training.
finetune/duplex_distill/launch_cluster.sh --run-dir /home/param/runs/pose-runs/duplex-distill-RUN --check-only

# Selected production configuration: two examples per GPU, 112 globally.
finetune/duplex_distill/launch_cluster.sh --run-dir /home/param/runs/pose-runs/duplex-distill-RUN --micro-batch-size 2 --text-chunk-size 512 --text-direct-addmm
finetune/duplex_distill/status_cluster.sh --run-dir /home/param/runs/pose-runs/duplex-distill-RUN --tail 10

# Full-depth six-step scratch profile, with the same resolved training implementation.
# Fresh production initialization, no W&B/evaluation/saves, automatic exit.
finetune/duplex_distill/launch_cluster.sh --run-dir /home/param/runs/smokes/duplex-distill-PROFILE --micro-batch-size 2 --text-chunk-size 512 --text-direct-addmm --profile-steps 6 --no-wandb
```

For a graceful stop, create `control/STOP_AFTER_SAVE` on any training host. Landsend-0 is now a training host, so this local command works there. The next completed update broadcasts the request, saves on all nodes, and exits:

```bash
touch /home/param/runs/pose-runs/duplex-distill-RUN/control/STOP_AFTER_SAVE
```

For immediate termination without a new checkpoint, stop only this launch’s recorded process trees:

```bash
python3 finetune/duplex_distill/cluster_ops.py stop --run-dir /home/param/runs/pose-runs/duplex-distill-RUN
```

## Checkpoints and resume

There are **no evaluations or step-based scheduled exports**. Rank zero requests a checkpoint after a completed update when 7,200 seconds have elapsed since training began or the last save finished. The final update and graceful stop also save.

Every node writes `/home/param/models/trained-weights/<run-name>/checkpoints/step-NNNNNN/`: FP32 model/optimizer/RNG state under `resume/`, BF16 weights under `export/`, and replicated decoder state once per node. After all ranks verify the new resume files, every node atomically commits `checkpoints/LATEST_RESUME.json`. Only after every node succeeds may older validated resume directories be removed. **Older BF16 exports remain.** Failed or questionable pruning retains files and reports a warning.

Globally synchronized local save errors use `CheckpointSaveError`. A failed periodic save preserves previous checkpoints, continues training, and schedules another attempt after 60 seconds. Final/stop saves make up to three attempts with five-second gaps, then fail visibly if none succeeds. Retries use distinct `step-NNNNNN-retry-AA` paths, preserving failed attempts for inspection. Transport failures and arbitrary exceptions propagate; they are not assumed recoverable. A latest-marker or pruning failure warns and keeps older resumes; the new verified checkpoint remains usable by its explicit path. The CPU fault-injection driver is `tests/distributed_checkpoint_faults.py`.

`export/NODE_COMPLETE` covers only that node’s subset, not all 56 ranks. Resume requires the same host/rank mapping, microbatch, and schedule, with corresponding local checkpoint files on every host. Read the latest marker to identify the actual completed step; pass its resume directory and original batch flags. W&B resumes the recorded ID with `resume="must"`, using `wandb.json` from the original or output run directory. Missing metadata, project mismatch, or conflicting IDs fails initialization across all ranks; explicitly use `--no-wandb` only when disabling logging is intended:

```bash
finetune/duplex_distill/launch_cluster.sh --run-dir /home/param/runs/pose-runs/duplex-distill-RESUMED --resume /home/param/models/trained-weights/duplex-distill-RUN/checkpoints/step-NNNNNN/resume --micro-batch-size 2 --text-chunk-size 512 --text-direct-addmm
```

For a portable export, gather `rank0.pt` through `rank55.pt` from all seven hosts plus matching `rep.pt` and `meta.json`. Verify source manifest hashes and preserve exact `vgain_theta_fp32` metadata. Only after complete verification should metadata change from `node-local-shards` and a global completion marker be written.

## Validation

The CPU suite covers balanced KL and decoder gradients, teacher isolation, production-loader equality, shared-frame sampling, complete epoch coverage, resume skipping, packed attention boundaries, mixed/all-dummy rows, and packed gradient normalization. Real-data checks run when the pinned data and model configuration are installed. Checkpoint tests cover exact optimizer/RNG restoration, retention safety, migrated model storage, and W&B identity recovery. Launcher tests verify complete source synchronization without contacting another host.

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m unittest discover -s finetune/duplex_distill/tests -v
# CPU data/FFmpeg verification on a host with the staged dataset.
.venv/bin/python -m finetune.duplex_distill.data --audio
# Local two-process CPU proof of DTensor save/resume and exact AdamW continuation.
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 -m finetune.duplex_distill.tests.distributed_checkpoint_check
```

Generated profiles, monitoring reports, and checkpoint-recovery artifacts are run outputs and are excluded from the training source package. The benchmark utilities at this package's top level remain source: `benchmark_text_head.py` also supplies the selected direct-addmm implementation used during training.
