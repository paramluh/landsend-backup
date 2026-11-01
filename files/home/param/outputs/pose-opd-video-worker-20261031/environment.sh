#!/usr/bin/env bash
# Existing tested video environment; no PPM training model dependencies.
source /home/param/vansh-video-models/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/param/outputs/pose-opd-video-worker-20261031/vendor/vansh-video-models:/home/param/pose-worktrees/pose-opd-live-demo/infer/pose_opd_live/renderer/amu${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/home/param/outputs/pose-opd-video-worker-20261031/cache/inductor
export TRITON_CACHE_DIR=/home/param/outputs/pose-opd-video-worker-20261031/cache/triton
export TORCH_EXTENSIONS_DIR=/home/param/outputs/pose-opd-video-worker-20261031/cache/extensions
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
