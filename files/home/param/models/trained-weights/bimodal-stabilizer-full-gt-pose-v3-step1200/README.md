# Full-GT-trained bimodal stabilizer — pose-v3, step 1200

Private training-state archive for `InternalCan/bimodal-stabilizer-full-gt-pose-v3-step1200`. The completed run contains **1,200 data rounds: 150 critic-only warmup rounds and 1,050 joint rounds**, yielding 1,050 student updates and 4,800 critic updates. The raw student and critic weights are saved; there is no EMA model or LoRA adapter.

Training uses one target per row with a GT sink and six prior GT latent slots. Both arms start the target at t=1 and use eight student integration steps (`[2, 2, 2, 2]`) to reach t=0.2. Independent history dropout probabilities `[0.0, 0.1, 0.2, 0.2]`, nearest predecessor first, remain fixed across a target rollout. Global batch size is 160 over 40 GPUs. At autoregressive inference, the arm name describes its training history: both students receive only the initial GT sink and then use generated history.

Student initialization and the frozen teacher use raw pose-v3 finetune step-200 weights, SHA-256 `7f986500452a4aceee5baac4ada2d5191600a82f25e126a942eb1abf19e55d85`. Those teacher weights, Mimi assets, VAE assets, training data, and the Python/CUDA environment are external dependencies and are not included in the support archive. For the published comparison, the student t=0.2 endpoint is passed unchanged to that teacher for ten resolver steps to t=0, without renoising.

## Files and identity

- `training.toml`: byte-for-byte frozen production config, including the original local paths.
- `provenance.json`: original checkpoint file sizes and hashes, node mapping, counters, strict compatibility signature, and source-bundle review.
- `frozen-training-source.tar.gz`: frozen `source-final-v1` source, with the unrelated credential-bearing `setup.sh` excluded. Included file contents are unchanged.
- `nodes/<hostname>/step-001200/`: the original node-local checkpoint files. Keep all five host directories separate.

| Host | Global ranks | Tensor files |
| --- | --- | --- |
| richmond-7 | 0–7 | student + critic + 8 rank shards |
| richmond-8 | 8–15 | 8 rank shards |
| richmond-9 | 16–23 | 8 rank shards |
| richmond-10 | 24–31 | 8 rank shards |
| richmond-11 | 32–39 | 8 rank shards |

Every node directory also contains its distinct `metadata.json` and `checkpoint.complete.json`. The checkpoint has **52 essential files**: two model files on `richmond-7`, forty globally numbered `rank-00000.pt` through `rank-00039.pt` files distributed eight per node, and ten metadata/completion files. Checkpoint IDs are deliberately node-local; do not merge or overwrite these metadata files.

The inference student is `nodes/richmond-7/step-001200/student.safetensors` (SHA-256 `1b5a07435cdc8cb2efa4f2a8b6ca75cfaeb86ab797ed656952fbd2e57cb9fc9b`). The critic is `nodes/richmond-7/step-001200/critic.safetensors` (SHA-256 `8ac4fa6a45d59f6d324c7014bba55a3b9a52182cfc84008ad07664d880fd391b`). Original run: `run-20261016-134249`; completed at `2026-10-16T16:14:48.672230+00:00`. [Training run](https://wandb.ai/canopy-labs/param-distillation/runs/i2ui9imf).

## Restoring the saved state

Use the frozen training code and recreate its environment and external assets. For an exact restore, put each host's own `step-001200` contents at the **same absolute restore path on that host**, retaining the original 40-rank ordering in the table. Use a fresh launcher tag/control directory and supply `--set 'resume_from="/absolute/restore/step-001200"'`. This setting is illustrative, not a standalone launcher command.

The frozen validator requires world size 40, the actual hostname/rank map, each node's local ranks, retained node-local optimizer shards, and the complete compatibility signature. This locks the original per-node paired-CSV byte hashes, data-distribution fingerprint, teacher identity, conditioning, batch size, and training schedule. Verify downloaded bytes against `provenance.json` before restoring. The original model files are needed on the master host; other hosts retain their own rank shards and metadata.

Each rank shard contains both roles' FP32 master parameters and AdamW moments/step, scheduler state, Torch/CUDA/Python/NumPy RNG state, and counters. There is no separate EMA, GradScaler, or dataloader-state file. The deterministic data sampler restarts at `counters.data_steps`. The scheduler lambda is not serialized (`lr_lambdas` is null); the frozen source and config recreate it before loading scheduler state.

**This completed checkpoint is not a turnkey extension to more training.** Its counters already equal `total_steps=1200`, so an unchanged strict resume performs zero additional updates. Increasing `total_steps` is rejected by the saved compatibility signature. A different schedule, host topology, or data distribution requires an explicit, separately validated migration. Preserving the optimizer state enables that future work but does not make such a migration automatic.

The historical `source-manifest.json` inside the source bundle describes the original snapshot, including the omitted setup script. The exclusions and included-file hashes in `provenance.json` describe this sanitized archive. Upload completion must be checked against the uploader's final remote verification; this document records the original training state.
