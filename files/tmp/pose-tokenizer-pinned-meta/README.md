---
license: mit
tags:
- keypoints
- model_hub_mixin
- pose-tokenizer
- pytorch_model_hub_mixin
- vqvae
---

# pose-tokenizer-32cb-200k-noshift

Causal pose tokenizer (32 codebooks, codebook size 2048, input_features=165 = 55 joints x {y,x,conf}).

- Run: `base-causal-pretrain-32cb-pose20k-subset1.2tb`, checkpoint **step 200000**.
- **`causal_loss_shift = 0`** (no latency budget / no lookahead): `recon[t]` reconstructs `gt[t]` from inputs `<= t`. True zero-latency causal reconstruction; no de-shift needed at inference.
- Target fps 25, segment_frames 128.

Loaded with the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin):
- Code: https://github.com/canopy-labs-internal/pose-tokenizer.git

`config.yaml` is the training config (kept for the reconstruction/inference scripts).
Optimizer state (`training_state.pt`) is intentionally omitted.
