#!/usr/bin/env bash
set -euo pipefail
DEPLOY=/home/param/pose-mono-dcp227
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export GENWEB="$DEPLOY/site" GENWEB_PEOPLE="$DEPLOY/prompts"
export HF_HUB_OFFLINE=1 POSE_MODELS="$DEPLOY/models"
export POSE_TOKENIZER_DIR="$DEPLOY/models/pose-tokenizer-32cb-200k-noshift"
export ELECTRON_ENCODER="$DEPLOY/models/electron-vae/encoder-v2-pose.safetensors"
export VIDEO_REPO="$DEPLOY/renderer/video-models" VIDEO_PYTHON="$DEPLOY/renderer/.venv/bin/python"
export VIDEO_CHECKPOINT="$DEPLOY/models/video-step2911/student.safetensors"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH=/usr/include/python3.12
export TORCHINDUCTOR_CACHE_DIR="$DEPLOY/runtime/cache/inductor" TRITON_CACHE_DIR="$DEPLOY/runtime/cache/triton"
exec "$VIDEO_PYTHON" -u "$DEPLOY/renderer/render_queue.py" "$@"
