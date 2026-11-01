#!/usr/bin/env bash
# Run from anywhere; paths are resolved relative to this script's directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"


sudo apt-get update
sudo apt-get install -y \
  ffmpeg \
  libavcodec60 \
  libavcodec-dev \
  libavformat60 \
  libavformat-dev \
  libavutil58 \
  libavutil-dev \
  libswresample4 \
  libswresample-dev \
  libswscale7 \
  libswscale-dev

pip install uv
uv sync --upgrade-package torchcodec

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail during setup if TorchCodec cannot find compatible FFmpeg/PyTorch libs.
uv run python - <<'PY'
from torchcodec.decoders import AudioDecoder

print(f"TorchCodec import OK: {AudioDecoder}")
PY

# Pre-warm the Mimi cache before subscribing. Set mimi_local_files_only: true
# in tokenization_config.yaml only if this cache is baked into the image.
uv run python - <<'PY'
from transformers import AutoFeatureExtractor, MimiModel

model_id = "kyutai/mimi"
AutoFeatureExtractor.from_pretrained(model_id)
MimiModel.from_pretrained(model_id)
PY

uv run python tokenize_speech.py