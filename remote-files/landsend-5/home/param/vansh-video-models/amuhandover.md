# Amu handover: latest unguided bimodal checkpoint

Use branch **`param/bimodal`** and the **step-2,911 student** from
[InternalCan/full-gt-nine-ref-endpoint-unguided-20261020](https://huggingface.co/InternalCan/full-gt-nine-ref-endpoint-unguided-20261020).
This replaces the earlier two-model, no-reference handover.

The same checkpoint supplies both stabilizer and resolver. Choose **raw** to
match the comparison videos, or explicitly choose EMA. Both need the actor's
**nine-slot reference bank**, pose-v3 latents, local-window Mimi codes, the
frozen Mimi codec, and the **Wan2.2 5B VAE**. No teacher, separate resolver,
LoRA merge, critic, optimizer state, or training launcher is needed.

Workflow: install dependencies → download one student and the two auxiliary
models → stage a prepared clip and actor references → generate a config →
check it → run one sampling command.

## 1. Checkout and environment

Use Linux, Python 3.12, and an allocated Hopper GPU (H100/H200), one process per
GPU, batch size 1. The backend imports FlashAttention 3 directly; A100/consumer
GPU compatibility is not established. Compilation and graph capture happen
on first use. No new GPU run was launched during this handover update.

```bash
git clone --branch param/bimodal --single-branch \
  https://github.com/canopy-labs-internal/vansh-video-models.git
cd vansh-video-models
export REPO="$PWD"
export HANDOVER="$HOME/outputs/amu-handover"
mkdir -p "$HANDOVER"
```

For an existing clone, fetch, switch to `param/bimodal`, and pull with
`--ff-only`. Reuse a working environment when available. These are the existing
handover's pinned inference versions. Install FFmpeg/ffprobe and Python 3.12
venv/development support through your package manager first; the CUDA
driver/toolchain must support the CUDA 13.0 wheels and compilation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt PyYAML==6.0.3 huggingface-hub==1.33.0
python -m pip install \
  'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl'
python -m pip install flash-attn-3==3.0.0 \
  --index-url https://download.pytorch.org/whl/cu130 --no-deps
ffmpeg -version
ffprobe -version
```

Both attention packages are needed: `flash_attn_interface` for video
self-attention and `flash_attn` for audio cross-attention. Keep the pinned
Transformers version; the loader uses Mimi's internal module structure.
Use these focused commands rather than the general `setup.sh`, which also
performs unrelated downloads and machine-specific authentication.

## 2. Download the weights

Authenticate with your own HF account/token with read access to the private
InternalCan repository. Resolve 401/403/404 access errors before continuing;
do not substitute a different checkpoint.

```bash
hf auth login
hf auth whoami
export STUDENT_REV=49de8772d2ac9a624f557fd3518de4254474c63a

hf download InternalCan/full-gt-nine-ref-endpoint-unguided-20261020 \
  step-002911/student.safetensors README.md export-manifest.json checksums.sha256 \
  --revision "$STUDENT_REV" \
  --local-dir "$HANDOVER/checkpoints/full-gt"
```

Only download EMA if you want to sample that alternative:

```bash
hf download InternalCan/full-gt-nine-ref-endpoint-unguided-20261020 \
  step-002911/model_ema.safetensors --revision "$STUDENT_REV" \
  --local-dir "$HANDOVER/checkpoints/full-gt"
```

| Variant | Filename under `step-002911/` | Bytes | SHA256 |
| --- | --- | ---: | --- |
| Raw, matches comparison videos | `student.safetensors` | 13,964,521,376 | `71fcbf4c467553eff0b2916e25cbd6c57031d7bd58913b87ba1e939b04963a71` |
| EMA, optional | `model_ema.safetensors` | 12,988,285,752 | `2cbcc06de65ecb856c546f698924ac807fd94f4e48a1452ff2b28bab152e4efa` |

Both are complete native BIMODAL state dictionaries with the same 582 tensor
names/shapes, including the reference embedding and learned audio modules.
Raw contains FP32 and BF16 tensors; EMA is saved in BF16. Preserve their bytes
and let the loader set inference dtypes. Source directory
`step-003061-d7752b2361f4` counts 150 critic warmup rounds plus **2,911 student
updates**. Teacher guidance during this training was 1.0.

Download the frozen Mimi codec and VAE. Only the VAE file is needed from Wan,
not base diffusion shards or T5 weights:

```bash
hf download kyutai/mimi config.json model.safetensors \
  --revision 89091b3e466eb6a9d11e537bf26b144f194978f7 \
  --local-dir "$HANDOVER/assets/mimi"

hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir "$HANDOVER/assets/wan22-5b-weights"

printf '%s  %s\n' \
  71fcbf4c467553eff0b2916e25cbd6c57031d7bd58913b87ba1e939b04963a71 \
  "$HANDOVER/checkpoints/full-gt/step-002911/student.safetensors" \
  bac7e85083dcded655d24eaadde7e6eea34c0da1b35fa2d284e641bd2b942a5e \
  "$HANDOVER/assets/mimi/model.safetensors" \
  20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36 \
  "$HANDOVER/assets/wan22-5b-weights/Wan2.2_VAE.pth" | sha256sum --check -
```

If EMA was downloaded, check its table hash too. The HF checksum file lists
both variants, so checking it wholesale requires both files. Raw plus auxiliary
models is about 17.17 GB; with EMA too it is about 30.16 GB. Allow additional
disk for the environment, compiler caches, inputs and outputs. The generated
config names the VAE explicitly; no repository symlink is needed.

## 3. Prepared clip and reference inputs

Supply these **already-preprocessed** files in one clip directory:

| Filename | Tensor key | On-disk shape / meaning |
| --- | --- | --- |
| `video_latents.safetensors` | `latents` | `[T,48,30,16]`; frame 0 is the clean appearance sink |
| `pose_latents_electron_v3.safetensors` | `latents` | `[T,48,30,16]`; aligned electron pose-v3 |
| `mimi_local.safetensors` | `codes` | Integer `[codebooks,audio_steps]`; typically `[32,3750]` for five minutes |

The demonstrated resolution is 256×480 at 25 RGB fps. Keep actor, crop, time
origin, and speaker/audio track aligned. `mimi_local` means the training's
**local-window Mimi encoding** (encoder attention window 250), not merely a
codec downloaded locally. A whole-chunk `mimi.safetensors` file cannot be
converted by renaming it. Preserve the original codebooks and full audio
timeline, even for a short sampling prefix.

Only video frame 0 supplies GT. Subsequent history comes from each mode's own
generated frames. The video file still needs its full timeline because usable
length is capped by it:

```text
usable_latents = min(video_T, pose_T, audio_steps // 2, configured_limit)
RGB_frames = 4 * usable_latents - 3
duration_seconds = RGB_frames / 25
```

1,875 usable latents produce 7,497 RGB frames / 299.88 seconds. A 33-latent
prefix produces 129 frames / 5.16 seconds.

References are **separate inputs**, not bundled in the HF checkpoint. The bank
path must contain the actor directory directly:

```text
reference-bank/
  <actor-uuid>/
    property_1/latents.safetensors
    ...
    property_9/latents.safetensors
```

Each file contains `video` shaped `[N,48,30,16]`: N already VAE-encoded images.
Use the validated 256×480 muon-encoded bank, made with the exact Wan VAE above.
Property order is front, left face, right face, mouth open, hands visible,
hands interacting, then three hair/clothing slots. One image is selected per
available property using the existing evaluation rule: NumPy RNG seed
`[1234, crc32(original CSV paths value) & 0x7fffffff]`. The sampling seed is
separate. Preserve that original `clip_key` when relocating a clip.

Missing individual properties use existing masked/pad-slot behavior; the actor
folder must exist and contain at least one valid property. Do not duplicate
other properties to fill gaps. Both modes receive the same selected references
and keep mask.

On the shared filesystem, the bank is currently at:

```bash
export BANK=/mnt/paramfs/param/outputs/reference-bank-richmond-strict/studio/reference-bank-256x480-muon-encoded/bank
```

For an off-cluster receiver, copy **only the requested actor's** property files
from an authorized machine with that mount, preserve this layout, and set
`BANK` to the local root. No whole corpus, whole bank, training CSV, or optimizer
data is required. Obtain the actor UUID and original CSV `paths` value together
with the prepared clip.

**Historical comparison inputs need restaging.** The old gallery's pose-v3
files are no longer at their recorded sharedFS paths; the old work-vm data
staging is also absent. Those historical paths are not ready samples. This
branch consumes prepared inputs; it does not regenerate raw-media pose-v3 or
local-Mimi features. Obtain matching prepared data from the data owner before
proceeding. The first gallery clip's provenance is:

```text
actor_id: cebaa7ca-9fbb-4244-b746-7f4d507681e2
clip_key: recordings/002702ede72fe17f6370442aaf8ccf9cdad634c0/speaker_1/chunks/000000
seed: 2026
pose-v3 SHA256: f83553e2145d18b6dbdb932182f7069f2d8e395d211de265e3b948feea430e13
```

Its nine reference files remain available (about 7.14 MB total). Full historical
provenance is in
`/mnt/paramfs/param/outputs/fullgt-inference-pose-v3-20261020/clips-selected.json`.
The gallery clips came from training data, not a held-out evaluation set.

## 4. Generate a config, check, sample

Once all inputs are present, set these to the chosen clip's actual values.
This example uses the first gallery clip **after restaging**:

```bash
export CLIP="$HANDOVER/data/clip-00"
export ACTOR=cebaa7ca-9fbb-4244-b746-7f4d507681e2
export CLIP_KEY=recordings/002702ede72fe17f6370442aaf8ccf9cdad634c0/speaker_1/chunks/000000

cd "$REPO"
source .venv/bin/activate
python -m inference.prepare_distilled \
  --root "$HANDOVER" --sample-dir "$CLIP" --reference-bank "$BANK" \
  --actor-id "$ACTOR" --clip-key "$CLIP_KEY" \
  --name clip-00 --weights raw --seed 2026 --latent-frames 1875

python -m inference.distilled "$HANDOVER/configs/clip-00-raw.json" --check
```

The helper writes an explicit JSON config (also accepted by the YAML reader)
and prints check/run commands. It expands paths and chooses the downloaded raw
or EMA file for **both modes**. Use `--weights ema` after downloading that
variant. Neither the helper nor `--check` launches CUDA, loads tensor payloads,
compiles, or generates video. They check paths, safetensors headers, reference
geometry/coverage and schedule. Header validation is not a content checksum or
an end-to-end GPU test; keep the hash checks above.

Use a fresh `--name` for each run; existing configs and output directories are
not overwritten. For a short first run, use `--name clip-00-preflight` and
`--latent-frames 33`. First-run compilation can dominate the short sample.

After allocating an available GPU, sampling is one command:

```bash
export TORCHINDUCTOR_CACHE_DIR="$HANDOVER/cache/inductor"
export TRITON_CACHE_DIR="$HANDOVER/cache/triton"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python -u -m inference.distilled \
  "$HANDOVER/configs/clip-00-raw.json"
```

Choose the allocated GPU index, not necessarily 0. No `torchrun`, NCCL, or
multi-node setup is required. For several clips, use separate configs/outputs
and one worker per allocated GPU. Compiler disk caches can be reused; each
process still loads weights and captures its CUDA graphs.

## 5. Sampling contract and outputs

| Component | Required behavior |
| --- | --- |
| Shared student | Load one raw or EMA checkpoint once; stabilizer/resolver are views of those weights |
| Stabilizer | Nine refs + sink + three history + four present slots; two Euler substeps in each of four bands, eight steps per generated frame, noise 1 → 0.2 |
| Resolver | Same nine refs + sink + two clean history + one present; one Euler step, 0.2 → 0 |
| Handoff | Consume the stabilizer endpoint unchanged; no renoising or new random draw |
| Guidance | Runtime audio guidance scale 1.0, full conditional audio |
| Execution | Compiled full-wavefront stabilizer with cached context and CUDA graphs; CUDA-graphed resolver, not a separate compile wrapper |
| Rendering | Existing mixed-precision model path, FP32 latent integration, bounded causal Wan VAE decode and Mimi audio muxing |

References are cached once per clip per mode; resolver history context is
rebuilt for each frame. These are the same backends and reference-selection
rule used for the 16 full-length raw-checkpoint gallery generations. This
portable entrypoint update has CPU checks; it has not rerun those GPU
generations or established new timing/memory guarantees.

Each run writes `config.json`, `status.json`, `resolved_latents.safetensors`,
`stabilised_latents.safetensors`, `video.mp4`, and `bounded-decode.jsonl`.
Latents use key `latents`, shape `[48,L,30,16]` FP32; the stabilized trajectory
is still at noise 0.2. Both trajectories are saved before rendering. Check
`status.json` for `complete` and inspect the streams:

```bash
ffprobe -v error -show_entries stream=codec_type,width,height,r_frame_rate \
  -show_entries format=duration -of json "$HANDOVER/runs/clip-00-raw/video.mp4"
```

CPU contract checks:

```bash
python -m unittest distillation.test_distilled_inference
python -m inference.prepare_distilled --help
python -m inference.distilled --help
```

If a check fails, fix the missing asset or conditioning mismatch. Do not swap
in the old no-reference pair, mix raw/EMA between modes, enable guidance or
renoising, change the two-substeps-per-band schedule, or substitute another
VAE to make an incompatible setup run.
