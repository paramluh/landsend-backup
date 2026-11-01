---
tags:
- video-generation
- distillation
- bimodal
---
# Latest unguided full-GT nine-reference endpoint checkpoint

Raw and EMA student weights at **2,911 student updates**, saved from the unguided
training run on 2026-10-20. Teacher guidance scale: **1.0**.

| Weights | File |
| --- | --- |
| Raw student | [student.safetensors](step-002911/student.safetensors) |
| EMA student | [model_ema.safetensors](step-002911/model_ema.safetensors) |

The source checkpoint is `step-003061-d7752b2361f4`: source round 3,061 includes
150 initial critic warmup rounds. The export folder `step-002911` counts student
updates only.

These are native BIMODAL model state dictionaries, preserving the exact saved
tensor names and dtypes. Raw weights contain 121 FP32 and 461 BF16 tensors; EMA
contains the same 582 parameter names and shapes, all BF16. Use the repository's
existing inference loader; this is not a Transformers pretrained layout.

Only this latest unguided checkpoint is included. No older checkpoints, guided
checkpoint, critic weights, or optimizer state are included. Original local
checkpoints remain intact. `export-manifest.json` records the checkpoint identity
and file sizes; `checksums.sha256` contains verified whole-file SHA256 hashes.
