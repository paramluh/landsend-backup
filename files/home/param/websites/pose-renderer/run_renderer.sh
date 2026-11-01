#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 POSE_TOKENIZER_DIR=/home/param/models/pose-tokenizer-32cb-200k-noshift
export ELECTRON_ENCODER=/home/param/models/electron-vae/encoder-v2-pose.safetensors
export CUDA_HOME=/usr/local/cuda
export CPATH=/home/param/pose-repo/.venv/include/python3.12:/home/param/pose-repo/.venv/include
export TORCHINDUCTOR_CACHE_DIR=/home/param/.cache/pose-sites-render/inductor TRITON_CACHE_DIR=/home/param/.cache/pose-sites-render/triton
exec /home/param/vansh-video-models/.venv/bin/python -u /home/param/websites/pose-renderer/render_queue.py "$@"
