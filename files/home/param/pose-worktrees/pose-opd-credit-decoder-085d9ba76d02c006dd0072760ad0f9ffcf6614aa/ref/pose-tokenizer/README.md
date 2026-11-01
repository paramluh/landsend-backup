# Pose Tokenizer

A VQ-VAE that tokenises body-pose keypoints into discrete codes. The architecture uses a temporal convolutional network (TCN) encoder/decoder with Residual Vector Quantization (RVQ), ported from [FaceTokenizer](https://github.com/canopy-labs-internal/face-tokenizer).

The model takes keypoint sequences (face + upper body + hands, 55 joints), converts them to a scale-invariant kinematic-chain representation, and compresses them into discrete tokens suitable for downstream generation tasks.

## Pose estimation model

We use **RTMW-X** (from [rtmlib](https://github.com/Tau-J/rtmlib)) as the pose estimation backbone. After evaluating several alternatives, RTMW-X was chosen for the following reasons:

| Model | Pros | Cons |
|---|---|---|
| **RTMW-X** | Whole-body (133 COCO joints incl. hands), fast ONNX inference, good hand tracking, no external detector needed (built-in person det) | Slightly lower body accuracy than ViTPose on edge cases |
| ViTPose | High body accuracy, well-studied | Slow, requires separate person detector (e.g. YOLOv8), poor hand keypoints, complex multi-model pipeline |
| MediaPipe | Very fast, runs on CPU | Low accuracy on hands at distance, limited to single person, non-standard skeleton |

RTMW-X provides 133 COCO-WholeBody keypoints per person per frame. From these, we select 55 joints: 5 face keypoints, 8 upper-body keypoints, 21 left-hand keypoints, and 21 right-hand keypoints. Confidence-aware checkpoints use interleaved `(y, x, confidence)` features, so their input width is `55 * 3 = 165`.

## Installation

Requires Python 3.12+. The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install system deps + uv, then sync the Python environment
bash setup.sh
```

Or manually:

```bash
pip install uv
uv sync
```

## Quick start for the agent

If you are an AI agent and need the recommended tokenizer right now, start with:

```bash
uv run python scripts/smoke_test_agent_tokenizer.py
```

This uses `configs/agent_default_32cb_tokenizer.yaml`, which points to:
- `InternalCan/base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048`
- local fallback: `/mnt/som-fs/james-checkpoints/pose-tokenizer-james/32cb_size_2048/base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048/step_200000`

You can force a source explicitly:

```bash
# Force local checkpoint
uv run python scripts/smoke_test_agent_tokenizer.py --source local

# Force HF Hub (set HF_TOKEN if private access is needed)
HF_TOKEN=... uv run python scripts/smoke_test_agent_tokenizer.py --source hub
```

### Load a tokenizer

```python
import torch
from pose_tokenizer import (
    AGENT_DEFAULT_LOCAL_CHECKPOINT,
    AGENT_DEFAULT_TOKENIZER_REPO,
    PoseTokenizer,
)

# From Hugging Face Hub
tokenizer = PoseTokenizer.from_pretrained(
    AGENT_DEFAULT_TOKENIZER_REPO,
    device="cuda",
)

# Or from a local HF-format checkpoint directory
tokenizer = PoseTokenizer.from_pretrained(
    AGENT_DEFAULT_LOCAL_CHECKPOINT,
    device="cuda",
)
```

`PoseTokenizer.from_pretrained(...)` loads the model architecture from `config.json` and weights from `model.safetensors`. It does not load optimizer state.

### Encode and decode pose features

```python
import torch

# Confidence-aware checkpoints expect:
#   (B, T, 165) = 55 joints x (y, x, confidence)
# or unbatched:
#   (T, 165)
features = torch.randn(1, 80, tokenizer.config.input_features)

# Encode to discrete tokens (list of tensors, one per codebook)
codes = tokenizer.encode(features)
# len(codes) == tokenizer.config.n_codebooks
# codes[i].shape == (B, T_compressed)

# Decode back to reconstructed features
reconstructed = tokenizer.decode(codes)
# reconstructed.shape == (B, T_out, 165)

# For confidence-aware checkpoints:
recon = reconstructed.reshape(1, -1, 55, 3)
recon_xy = recon[..., :2]
recon_confidence = recon[..., 2].clamp(0, 1)
```

### Loading from a local checkpoint

```python
from pose_tokenizer import PoseTokenizer

tokenizer = PoseTokenizer.from_pretrained(
    "/mnt/som-fs/james-checkpoints/pose-tokenizer-james/32cb_size_2048/"
    "base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048/"
    "step_200000",
    device="cuda",
)
```

The checkpoint directory must contain at least:

```text
config.json
model.safetensors
```

### Using the low-level model directly

```python
from pose_tokenizer import PoseTokenizerModel, PoseTokenizerConfig

config = PoseTokenizerConfig(
    input_features=165,
    num_keypoints=55,
    keypoint_dim=2,
    n_codebooks=8,
    codebook_size=1024,
)
model = PoseTokenizerModel(config=config)

features = torch.randn(8, 80, 165)  # (B, T, F)
out = model(features)

out["reconstruction"]    # (B, T, 165)
out["codes"]             # list of (B, T_compressed) int tensors
out["commitment_loss"]   # scalar
out["codebook_loss"]     # scalar
```

## Data format

The dataset expects `.npz` files produced by the pose estimation pipeline, each containing:

| Key | Shape | Description |
|---|---|---|
| `poses` | `(T, 133, 3)` or `(T, P, 133, 3)` | Per-frame keypoints `(y, x, confidence)`. Single-person or multi-person. |
| `orig_h` | scalar | Original video height (optional if using `dims_csv`) |
| `orig_w` | scalar | Original video width (optional if using `dims_csv`) |

### Providing dimensions via CSV

If your `.npz` files don't contain `orig_h`/`orig_w` metadata, you can supply video dimensions via a CSV file (`dims_csv`). The CSV must have columns `clip_stem`, `width`, `height`:

```csv
clip_stem,width,height
7615522494455368982_00-00.00_to_00-04.70,1080,1920
7615522494455368982_00-04.70_to_00-14.07,1080,1920
```

The `clip_stem` is the `.npz` filename without the `_poses.npz` suffix.

### Loading a dataset

A manifest file (one relative `.npz` path per line) is the recommended way to index a large dataset:

```python
from pose_tokenizer.data import PoseDataset

ds = PoseDataset(
    data_dir="pose-data/",
    manifest="manifest.txt",
    dims_csv="scripts/clip_dims.csv",
)

sample = ds[0]
sample["keypoints"]     # (T, 110) or (T, 165) with include_confidence=True
sample["confidence"]    # (T, 55)  — per-joint confidence
sample["global_state"]  # (T, 3)   — (root_y, root_x, shoulder_width) in [0, 1]
```

The preprocessing pipeline automatically:
- Selects the highest-confidence person per frame (if multi-person)
- Extracts 55 joints (5 face + 8 upper body + 21 left hand + 21 right hand)
- Converts absolute positions to kinematic-chain offsets (relative to parent joints)
- Normalises by median shoulder width for scale invariance
- Skips clips where the detection is too small/broken

## Training

Training uses [Accelerate](https://huggingface.co/docs/accelerate) for distributed training and [Weights & Biases](https://wandb.ai) for experiment tracking.

### Pretraining plan (npz → npy, then 16cb & 32cb)

The current pretraining plan is a two-stage pipeline, then a matched pair of runs
that differ **only** in codebook count (16 vs 32).

**Step 1 — Convert compressed `.npz` poses to memmappable `.npy`.** Training reads
`.npy` via memmap so it can slice random fixed-length windows without decompressing
whole clips. Run the offline converter once per dataset (resumable; see
[`scripts/convert_poses_to_npy.py`](scripts/convert_poses_to_npy.py)):

```bash
uv run python scripts/convert_poses_to_npy.py \
  --src /mnt/somfs/pose_cond/pose_cond_npz \
  --dst /mnt/somfs/pose_cond/pose_cond_npy \
  --manifest manifest.txt --workers 16 --threads-per-worker 8 --prefilter-existing
```

**Step 2 — Train the two matched tokenizers.** Both configs are
confidence-aware (`input_features: 165`), so use `train_confidence.py`. Each run
uses **all 8 H100s on the node** (`--num_processes 8`); run them one after the other:

```bash
# 32 codebooks — all 8 GPUs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch --multi_gpu --num_processes 8 \
  train_confidence.py --config configs/base_causal_pretrain_32cb.yaml

# 16 codebooks — all 8 GPUs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch --multi_gpu --num_processes 8 \
  train_confidence.py --config configs/base_causal_pretrain_16cb.yaml
```

With `batch_size: 128` per GPU and 8 GPUs, the effective batch is `128 × 8 = 1024`
(`gradient_accumulation_steps: 1`). Adjust `batch_size` down if a single GPU OOMs at
`segment_frames: 128`, and/or raise `gradient_accumulation_steps` to hold the
effective batch.

`configs/base_causal_pretrain_16cb.yaml` and `configs/base_causal_pretrain_32cb.yaml`
are **byte-for-byte identical except**: `n_codebooks` (16 vs 32), the matching
`vq_strides` list length (16 vs 32), the `wandb_run_name`
(`base-causal-pretrain-16cb` / `-32cb`), and the header comment. Everything else is
shared, so the two runs are a clean codebook-count ablation:

- `causal: true` with minimal-latency `-(s-1)` conv padding and `causal_loss_shift: 0`
- fixed frame-0 root (`first_frame_root: true`), confidence baked in (`input_features: 165`)
- `segment_frames: 128` (5.12 s @ 25 fps), 2× downsample, `codebook_size: 2048`
- identical optimizer/schedule, loss weights, and RTMW-style denoising corruption

Both configs are **multi-dataset ready**: their `datasets:` list has one active entry
plus a commented template — add more `data_dir`/`manifest`/`dims_csv`/`source_fps`
entries to pool datasets (each self-contained, composed via `ConcatDataset`).

### Configuration

All training parameters live in YAML files under `configs/` (e.g. `configs/wide_l2norm.yaml`):

```yaml
# Data
data_dir: pose-data/
manifest: manifest.txt
dims_csv: scripts/clip_dims.csv    # optional, provides orig_h/orig_w externally
num_files: 0                       # 0 = use entire manifest
val_count: 10000

# Training
epochs: 100
batch_size: 256
lr: 3.0e-4
reconstruction_weight: 1.0
commitment_weight: 0.25
codebook_weight: 1.0
num_workers: 10

# Eval / Checkpointing (step-based)
validation_interval: 500
save_interval: 5000
output_dir: checkpoints

# Wandb
wandb_project: pose-tokenizer
wandb_run_name: "my-experiment"
log_interval: 10
```

### Launch training

```bash
# Single GPU
accelerate launch train.py --config configs/wide_l2norm.yaml

# Override the wandb run name
accelerate launch train.py --config configs/wide_l2norm.yaml --run-name my-experiment

# Multi-GPU
accelerate launch --multi_gpu train.py --config configs/wide_l2norm.yaml

# Multi-GPU on GCE L4 instances (no NVLink/P2P)
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch train.py --config configs/wide_l2norm.yaml
```

For training with the additive relative-L1 chain loss, use `train_chain_loss.py`:

```bash
accelerate launch --multi_gpu train_chain_loss.py --config configs/wide_l2norm_rel_l1.yaml
```

### Resume from checkpoint

```bash
accelerate launch train.py --checkpoint checkpoints/my-experiment/step_25000
```

### Checkpoint structure

Checkpoints are organised by wandb run name, with `step_*` subdirectories:

```
checkpoints/
  my-experiment/                    # wandb run name
    config.yaml                     # training config snapshot
    step_145000/
      config.json                   # HF-format model config used by from_pretrained
      model.safetensors             # HF-format model weights used by from_pretrained
      training_state.pt             # trainer metadata, and optionally optimizer state
      README.md                     # generated model card stub
      model_config.json             # architecture config copy
      model_summary.txt             # parameter counts
    step_150000/
      ...
    step_200000/
      ...
```

For inference or Hub sharing, only `config.json` and `model.safetensors` are required. `training_state.pt` can be very large because it may include AdamW optimizer state; it is only needed for exact training resume and should not be uploaded to inference-only Hub repos.

## HuggingFace Hub

The model integrates with Hugging Face Hub via `PyTorchModelHubMixin`. The recommended API for users is the high-level `PoseTokenizer` wrapper:

```python
from pose_tokenizer import PoseTokenizer

# Download and load
tokenizer = PoseTokenizer.from_pretrained(
    "InternalCan/base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048",
    device="cuda",
)

# Save locally in HF format
tokenizer.save_pretrained("./base-causal-2x-wide-32cb-step200000")

# Push to hub
tokenizer.push_to_hub(
    "InternalCan/base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048",
    private=True,
)
```

You can also use the low-level model class directly:

```python
from pose_tokenizer import PoseTokenizerModel

model = PoseTokenizerModel.from_pretrained(
    "InternalCan/base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048"
)
```

If you push from a loaded `PoseTokenizer` or `PoseTokenizerModel`, only the model artifacts managed by the Hugging Face mixin are uploaded. If you upload files manually, exclude training-only files such as:

```text
training_state.pt
optimizer.pt
checkpoint.pt
```

## Architecture

| Component | Description |
|---|---|
| **Encoder** | Causal 1D convolutions with residual blocks and progressive strided downsampling |
| **Decoder** | Mirror of the encoder with nearest-neighbour upsampling |
| **Quantizer** | Residual Vector Quantization (RVQ) with configurable number of codebooks, optional rotation trick for improved gradient flow |

Current confidence-aware 8-codebook checkpoints use 8 codebooks, 1024 codes each, a final encoder-width latent, 2048-dim intermediate width, and 4x temporal downsampling. Older configs may use fewer larger codebooks.

### Key config parameters

| Parameter | Default | Description |
|---|---|---|
| `n_codebooks` | 8 | Number of RVQ codebooks |
| `codebook_size` | 1024 | Codes per codebook |
| `codebook_dim` | 16 | Dimension of each code vector |
| `d_model` | 2048 | Intermediate channel width |
| `num_blocks` | 3 | Encoder/decoder depth |
| `downsampling_rates` | `[1, 2, 2]` | Per-block temporal stride (product = 4x) |
| `kernel_size` | `[3, 3, 3]` | Per-block convolution kernel size |
