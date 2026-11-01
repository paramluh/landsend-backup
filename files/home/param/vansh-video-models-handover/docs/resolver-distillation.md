# One step resolver distillation

Train a dense resolver to take a ground-truth frame noised to `t=0.2` and predict its clean latent in one Euler step. The active experiment uses Landsend 0–4, forty H100 GPUs, and the final raw pose-v3 teacher checkpoint. It starts from the teacher with fresh optimizer state; it does not resume the canceled Richmond run. The stabilizer-dependent experiment was canceled after the stabilizer results were reviewed.

## Training objective

For a target latent `x0`, sample independent Gaussian noise `epsilon` and construct `x02 = 0.8*x0 + 0.2*epsilon`. The student produces `y = x02 - 0.2*v_student(x02, 0.2)`. Each target sees the initial clean sink and its two preceding GT history frames, subject to the fine-tuning history dropout described below. Generated outputs never become another target's history.

The frozen teacher and trainable flow-matching critic run only in resolving mode. Sample scoring time uniformly over `[0.02, 0.2]`, noise the detached generated output as `xt = (1-t)*y + t*epsilon`, and train the critic against `epsilon-y`. The student uses the existing velocity DMD gradient `(v_teacher-v_critic)/mean(abs(v_teacher-v_star))`, with a per-target normalizer and epsilon `1e-8`. There is no stabilizer rollout, additional renoising stage, or multi-step resolver teacher trajectory in this training loop.

Student, critic, and frozen teacher initialize from `/home/param/richmond-assets/teacher-pose-v3-step200/model.safetensors`, SHA256 `7f986500452a4aceee5baac4ada2d5191600a82f25e126a942eb1abf19e55d85`. Conditioning uses `pose_latents_electron_v3.safetensors` and the current `mimi_local.safetensors` corpus. The frozen Mimi frontend remains unchanged.

## Batching and memory

Every student update and every critic update uses exactly **160 target frames globally**, or four targets per GPU. Each outer step first generates sixteen outputs per GPU without gradients in one batched student forward. This pool supplies four sequential critic updates, each with global batch 160. The student then updates on a separate global batch of 160.

Four consecutive targets share their sink and overlapping GT history computations: the packed pass contains four present frames, five distinct history frames, and one sink. Each target still attends only to its own four-frame resolver window. This reduces repeated context work compared with four separately evaluated windows. Tests compare packed predictions and gradients with independent windows and verify that one generated target cannot affect another target's history.

The initial candidate uses no activation checkpointing. Student and critic weights are replicated, with FP32 optimizer masters and AdamW moments sharded across all forty ranks using the existing ZeRO-1 implementation. Only the active model's gradient flat is allocated. The frozen teacher's transformer blocks are sharded across the eight GPUs within each node and gathered over NVLink during its forward.

## History dropout

Use the exact parent-window probabilities from `experiments/bimodal/aloe-bimodal-studio-pose-v3-ft-200.toml`: `[0.1, 0.1, 0.0]`. These mean 10% probability of dropping the oldest of three parent history frames, 10% of dropping the oldest two, and 80% of keeping them all. Dropped frames use the learned padding token, with their pose contribution disabled; the sink remains valid.

The fine-tuning resolver inherits the parent mask through `BimodalWindows`; it does not independently resample dropout for its two history slots. Distillation preserves that inheritance. For full four-target windows, only the first resolver target can lose its oldest history frame, in 10% of parent windows: 2.5% of resolver targets overall. This is the full-window dropout behavior of the original fine-tuning pipeline. The dataset retains six preceding GT frames for its common row format; dropout applies only to the final three parent history positions.

## Schedule and outputs

Run 2,000 outer steps total: 150 critic-only warmup steps followed by 1,850 student updates. Each outer step has four critic updates, for 8,000 critic updates in total. Both optimizers use dense AdamW with peak LR `2e-6`, betas `(0.9, 0.99)`, weight decay `1e-4`, and gradient clipping at 2.0. The student warms up for 50 of its updates; the critic warms up for 100 of its updates. Each then follows a cosine decay so the LR actually used for its final update is `2e-7`, exactly one tenth of peak LR.

Save raw student and critic weights every 200 total steps, including step 2000. Each new checkpoint includes optimizer masters, Adam state, scheduler state, RNG state, counters, and model/data/config provenance. Publish its completion marker before pruning optimizer shards from the previous checkpoint. Retain optimizer state only in the latest complete checkpoint. Archive verified older weight checkpoints to a second allocated node when needed to fit available disk space; keep the latest resumable checkpoint on the master.

W&B project `param-distillation` receives global step, warmup/joint phase, losses, per-target DMD statistics, gradient norms, actual/next LRs, teacher-versus-critic velocity cosine similarity and RMS differences, history dropout counts, phase timings, and peak memory. The run directory also retains `metrics.jsonl`, configuration, checkpoint metadata, and a final completion record.

## Entry points

The Landsend production configuration is `experiments/distillation/resolver-gt-landsend.toml`; `resolver-gt.toml` retains the original Richmond paths. Launch the pushed Git revision with:

```bash
bash run_dmd_resolver.sh \
  'landsend-0 landsend-1 landsend-2 landsend-3 landsend-4' \
  experiments/distillation/resolver-gt-landsend.toml
```

`distillation/benchmark_resolver.py` runs five-node benchmarks and records exact source hashes, batch units, per-phase timing, memory, and exclusions. Start with the full pool batch and no checkpointing; use the memory frontier only after a confirmed CUDA OOM. Steady-step projections exclude compilation, critic warmup, and checkpoint writes. The final dropout-enabled benchmark measured a 1.516-second median and 63.714 GiB peak, with all losses and gradients finite.

Production outputs and deployment receipts live under `/mnt/paramfs/param/outputs/resolver-landsend-20261016`. The first 150 logged steps are critic-only warmup. The canceled Richmond run artifacts were deleted; benchmark evidence remains for reproducibility.
