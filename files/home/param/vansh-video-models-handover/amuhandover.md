# Amu handover: fast distilled pose-v3 inference

## Instructions for the receiving agent

Use branch `param/amu-handover`, branched from `param/bimodal` at `8ce07ea`.
**Pull the full-GT distilled stabilizer and the GT distilled resolver from
Hugging Face, using the exact revisions and files below.** Run the committed
`python -m inference.distilled CONFIG.yaml` entrypoint, not a training launcher.
Keep downloads, inputs, configs, caches, logs, and generated outputs under
`~/outputs/amu-handover`. Confirm access to an available, authorized GPU before
launching; do not stop somebody else's jobs to make room.

This is the pair used for the earlier **full-GT** comparison videos. It is not
`vanshjjw/aloe-bimodal-studio-9-sinks`, the later reference-bank/audio-guidance
experiment. Neither a reference bank nor audio guidance is needed here.

## What the fast path actually does

| Component | Required inference settings |
| --- | --- |
| Stabilizer | 1 sink + 3 history + 4 present slots; 2 Euler substeps per slot, **8 total steps per frame**, from noise level 1 to 0.2 |
| Resolver | 1 sink + 2 clean history + 1 present; **1 Euler step** from 0.2 to 0 |
| Handoff | Pass the stabilizer's committed tensor unchanged to the resolver; no renoising, rescaling, or extra random draw |
| Stabilizer backend | `WavefrontNet`: all four present slots in one full-wavefront pass; cached sink/history K/V; `torch.compile`; CUDA graphs |
| Resolver backend | `GraphResolver`: CUDA graph covering clean-context construction and the one-step update; separate cached audio encoding |
| Precision | Existing BF16 model/autocast path and FP32 latent integration; do not globally cast everything to BF16 |
| Rendering | Bounded-memory causal Wan VAE decode, then H.264 MP4 with Mimi-decoded audio |

The resolver is **CUDA-graphed, not separately `torch.compile`-wrapped**. Its
clean context is rebuilt for each new frame; it is not a persistent cross-frame
history K/V ring like the stabilizer. The two models have different learned
audio encoders, so the runner deliberately does not share their audio tokens.

This backend already produced the five-minute comparison videos. The handover
entrypoint packages that orchestration into the repository instead of depending
on the original untracked script under `~/outputs`. The generic
`bimodal/sampler.py` CLI uses the non-graph resolver and full-video decoding;
`distillation/render_long.py` assumes a ten-step teacher resolver. Neither is
the recommended entrypoint for this pair.

Compilation, graph capture, model loading, and rendering are not steady-state
sampling. Existing five-minute jobs recorded roughly 127–223 seconds of sampling
after initial capture and about 35 GiB peak allocation in several runs; another
job reached about 43 GiB. These are historical measurements, **not a latency or
minimum-memory guarantee** for another machine. The newly packaged CLI has CPU
contract tests; it has not had a separate end-to-end GPU run during this handover.

## 1. Environment

Use Linux, Python 3.12, and an NVIDIA Hopper GPU. H100 80 GB and H200 are the
validated hardware family; A100/consumer GPU compatibility is not established
because this code imports FlashAttention 3 directly. Start with one process on
one GPU and batch size 1. Have adequate host RAM for loading two approximately
14 GB checkpoints, and at least 40 GB of free disk for assets plus per-clip
inputs/outputs. A five-minute pair of saved FP32 latent tensors is about 346 MB.

From an existing clone:

```bash
git fetch origin
git switch param/amu-handover
export REPO="$PWD"
export HANDOVER="$HOME/outputs/amu-handover"
mkdir -p "$HANDOVER"
```

Reuse the working environment if available. Otherwise, the versions below match
the observed environment. Install FFmpeg/ffprobe and Python 3.12 venv/development
support through your machine administrator or package manager first. The CUDA
driver/toolchain must support the CUDA 13.0 PyTorch wheels and compilation.
Do **not** run `setup.sh` blindly: it includes machine-specific authentication
and downloads assets that this handover does not need. Authenticate using your
own account, never a token embedded in a script.

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

Both attention packages are required: `flash_attn_interface` for video
self-attention and `flash_attn` for audio cross-attention. Keep the pinned
Transformers version from `requirements.txt`; the checkpoint loader uses Mimi's
module structure directly.

## 2. Download exactly these weights

The InternalCan repositories require an HF account/token with access. Being
logged in as `vanshjjw` is not inherently required; repository permission is.
If a download returns 401/403/404, verify identity and repository access instead
of substituting another model. The commands download only inference weights
and small provenance files, not critics or optimizer/rank shards.

| Role | HF repository | Revision | Weight filename |
| --- | --- | --- | --- |
| Full-GT stabilizer | `InternalCan/bimodal-stabilizer-full-gt-pose-v3-step1200` | `af6806541a1fce20ec8c52cd7d6ab1a523eecb03` | `nodes/richmond-7/step-001200/student.safetensors` |
| GT resolver | `InternalCan/bimodal-resolver-gt-pose-v3-step2000` | `0e82bbf07c6198a9eb06b38a19ed58bf6a04c2c5` | `student.safetensors` |

Both files are full dense student checkpoints, each 13,961,556,688 bytes with
580 tensors. They are not LoRA adapters: **no teacher/base checkpoint, merge,
critic, EMA selection, or training-state restoration is needed**. Although the
stabilizer HF model card describes an older teacher-resolver comparison, this
handover deliberately pairs it with the one-step distilled resolver above.

```bash
hf auth login
hf auth whoami

hf download InternalCan/bimodal-stabilizer-full-gt-pose-v3-step1200 \
  nodes/richmond-7/step-001200/student.safetensors \
  nodes/richmond-7/step-001200/metadata.json training.toml README.md \
  --revision af6806541a1fce20ec8c52cd7d6ab1a523eecb03 \
  --local-dir "$HANDOVER/checkpoints/stabilizer-full-gt"

hf download InternalCan/bimodal-resolver-gt-pose-v3-step2000 \
  student.safetensors \
  --revision 0e82bbf07c6198a9eb06b38a19ed58bf6a04c2c5 \
  --local-dir "$HANDOVER/checkpoints/resolver"

export STABILIZER="$HANDOVER/checkpoints/stabilizer-full-gt/nodes/richmond-7/step-001200/student.safetensors"
export RESOLVER="$HANDOVER/checkpoints/resolver/student.safetensors"
printf '%s  %s\n' \
  1b5a07435cdc8cb2efa4f2a8b6ca75cfaeb86ab797ed656952fbd2e57cb9fc9b "$STABILIZER" \
  cb3e1c26b5edacaba03e91921c6cbe68c49b72b4343366cb7c906c09e8fbc743 "$RESOLVER" \
  | sha256sum --check -
```

Two auxiliary assets are required. Mimi is the frozen codec, while its trained
conditioning encoder/adapter weights are already in each student checkpoint.
Only the Wan VAE is needed, not Wan's base diffusion shards or T5 weights.

```bash
hf download kyutai/mimi config.json model.safetensors \
  --revision 89091b3e466eb6a9d11e537bf26b144f194978f7 \
  --local-dir "$HANDOVER/assets/mimi"

hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir "$HANDOVER/assets/wan22-5b-weights"

printf '%s  %s\n' \
  20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36 \
  "$HANDOVER/assets/wan22-5b-weights/Wan2.2_VAE.pth" | sha256sum --check -
```

The loader expects `Wan2.2_VAE.pth` at
`$REPO/wan22-5b-weights/Wan2.2_VAE.pth`. In a fresh clone, link the downloaded
directory there:

```bash
ln -s "$HANDOVER/assets/wan22-5b-weights" "$REPO/wan22-5b-weights"
```

If that repository path already exists, **do not overwrite it**; verify its
`Wan2.2_VAE.pth` against the same hash and retain it. All actual downloaded
weights remain outside Git. Once assets are present, `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` can be used for inference.

## 3. Input data contract: pose-v3 and local Mimi

Supply a directory per clip, not a raw MP4/WAV, pose JSON, or rendered skeleton:

```text
clip-00/
  video_latents.safetensors
  pose_latents_electron_v3.safetensors
  mimi_local.safetensors
```

| File | Tensor key | On-disk shape / meaning |
| --- | --- | --- |
| `video_latents.safetensors` | `latents` | `[T, 48, 30, 16]`, normally BF16, Wan video latents; frame 0 is the clean identity/appearance sink |
| `pose_latents_electron_v3.safetensors` | `latents` | `[T, 48, 30, 16]`, normally BF16, aligned electron **pose-v3** latents |
| `mimi_local.safetensors` | `codes` | `[codebooks, audio_steps]`, integer codes; existing clips use int16 `[32, 3750]` for five minutes |

These are 256-pixel-wide by 480-pixel-high clips at 25 RGB frames/sec. Wan's
latent cadence is approximately 6.25 Hz; Mimi codes are 12.5 Hz from 24 kHz
audio. Use the same crop, time origin, speaker track, and synchronization across
all three files. The model conditions on the first 16 codebooks; retain the
original 32-codebook files for the matching waveform decode.

**Local Mimi has two distinct meanings here:**

1. `audio.model_name` points to the downloaded local Mimi model directory, so
   inference does not depend on a remote download.
2. `mimi_local.safetensors` must contain the training-compatible **local-window
   Mimi encodings** (encoder attention window 250 steps). A whole-chunk
   `mimi.safetensors` file is not interchangeable. Renaming it or merely loading
   the codec from a local directory does not convert its encoding semantics.

Do not substitute pose-v1/v2 filenames or regenerate arbitrary pose/audio
features and assume compatibility. This repository's inference entrypoint
consumes already-preprocessed tensors; it does not implement the raw-media
pose-v3 and local-Mimi dataset preparation pipeline. For new media, obtain the
matching preprocessing pipeline/artifacts from the data owner first. Ordinary
`MimiModel.encode` without validating the encoder-window behavior is not a
documented substitute.

The loader permutes video and pose into `[48, T, 30, 16]` and uses:

```text
usable_latent_frames = min(video_T, pose_T, audio_steps // 2, configured_cap)
decoded_video_frames = 4 * usable_latent_frames - 3
duration_seconds = decoded_video_frames / 25
```

Our source files have 1,876 video/pose latents and 3,750 audio codes, giving
1,875 usable latents, 7,497 RGB frames, and **299.88 seconds**. The minimum-length
check prevents accidentally accepting a short clip. Keep the complete audio
codes even for a short smoke run: the sliding audio tables preserve the
training context and need more than just the audible prefix.

Only video latent frame 0 is used as ground truth by generation. Later video
latents are not supplied as history: stabilizer history comes from its own
committed frames; resolver history comes from its own resolved frames.
**Full-GT describes the stabilizer's training regime, not teacher forcing at
inference.** The loader still requires a video-latent file with the full
timeline length; passing only a one-frame file truncates the run.

For the exact existing comparison set, ask Param for the staged files and
`/home/param/outputs/pose-v3-comparison-20261017/data/clips_manifest.json`.
Clip 00 is under that root's `data/clip`; clips 01–15 are under
`data/clips/clip-01` through `clip-15`. Seeds are 2026 through 2041. Those are
private local/cluster paths, **not downloadable assets included in this Git
branch or the two model repositories**. The selected clips came from the
training shard, not a held-out evaluation set. No reference-bank transfer is
needed for this handover.

## 4. Configure and run one clip

The following shell block expands paths before writing YAML; the runner does
not expand `$HOME`, environment variables, or `~` inside YAML. Put the three
approved input files in `$HANDOVER/data/clip-00` first. Use a fresh output
directory for every invocation; the runner refuses to overwrite an existing
one.

```bash
mkdir -p "$HANDOVER/configs"
cat > "$HANDOVER/configs/clip-00.yaml" <<EOF
sample_dir: "$HANDOVER/data/clip-00"
output_dir: "$HANDOVER/runs/clip-00-full-gt"
pose_file: pose_latents_electron_v3.safetensors
mimi_file: mimi_local.safetensors
type: universal_wide_bimodal
seed: 2026
batch_size: 1
max_latent_frames: 1875
minimum_latent_frames: 1875
fps: 25
compile: true
cuda_graph: true
wavefront: true
renoising: false
stabilising:
  checkpoint_path: "$STABILIZER"
  num_history: 3
  num_present: 4
  history_t: 0.2
  substeps: 2
resolving:
  checkpoint_path: "$RESOLVER"
  num_history: 2
  num_present: 1
  t_start: 0.2
  num_steps: 1
audio:
  model_name: "$HANDOVER/assets/mimi"
  in_dim: 512
  num_codebooks: 16
  num_layers: 4
  cross_attention_codes: 4
  drop: 2
  injection_offset: 4
  injection_interval: 3
  max_blocks: 8
EOF

cd "$REPO"
source .venv/bin/activate
export TORCHINDUCTOR_CACHE_DIR="$HANDOVER/cache/inductor"
export TRITON_CACHE_DIR="$HANDOVER/cache/triton"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python -u -m inference.distilled \
  "$HANDOVER/configs/clip-00.yaml"
```

For an explicitly approved short preflight, copy the config to a new filename,
set **both** `max_latent_frames` and `minimum_latent_frames` to 33, and change
`output_dir` to a fresh `.../runs/clip-00-preflight`. That produces 129 RGB
frames, about 5.16 seconds, exercising capture, latent saving, decoding, and
audio muxing. First-run compilation can dominate such a short test. Restore
both frame limits to 1875 for the five-minute run; never publish the preflight
as a full-length result.

## 5. Outputs, checks, and parallelism

Each output directory contains:

- `config.json`: the effective configuration.
- `status.json`: loading/sampling/saving/decoding/complete or failed, progress,
  timings, and peak GPU allocation.
- `resolved_latents.safetensors`: final clean generated video, key `latents`,
  `[48, L, 30, 16]` FP32. This layout differs from input files.
- `stabilised_latents.safetensors`: the intermediate noise-level-0.2 trajectory,
  same shape; not the final clean video.
- `video.mp4`: final resolved H.264 video with decoded conditioning audio.
- `bounded-decode.jsonl`: rendering receipt.

Both latent files are saved before rendering, so a decode failure does not
discard the completed model outputs. The runner checks finiteness on export
and probes the final video's duration. Inspect `status.json` for `complete`
and confirm a full run has 1,875 latent frames / 7,497 decoded frames. To inspect
the encoded streams:

```bash
ffprobe -v error -show_entries stream=codec_type,width,height,r_frame_rate \
  -show_entries format=duration -of json \
  "$HANDOVER/runs/clip-00-full-gt/video.mp4"
```

For additional clips, use one config per clip with its own `sample_dir`, seed,
and fresh `output_dir`. Run one worker per explicitly allocated GPU using
`CUDA_VISIBLE_DEVICES`; this is independent inference, not distributed
training, and does not require `torchrun`, NCCL, or 40/48 GPUs. Do not put two
workers on the same GPU. Repeated processes can reuse compiler disk caches but
must still load weights and capture their own CUDA graphs. This minimal CLI
handles one clip per invocation, not an in-process multi-clip queue.

Do not silently disable compilation, switch off full-wavefront sampling,
increase stabilizer substeps to 8 per slot, select a ten-step teacher resolver,
enable renoising, or use the unrelated reference-conditioned model to work
around an error. Diagnose the asset/version/config mismatch instead.

CPU-only contract tests for the packaged entrypoint:

```bash
python -m unittest distillation.test_distilled_inference
python -m inference.distilled --help
```
