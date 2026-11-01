#!/bin/bash
export NEXT_OPD_ROOT=/home/param/pose-runs/next-opd-demo-20261030
export PPM_FA3_ROOT=/home/param/pose-repo/.cache/kernels/vllm-flash-attn3
export CPATH="/home/param/pose-repo/.venv/include/python3.12:/home/param/pose-repo/.venv/include${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PPM_CD=2 GEMM_PRECISION=bf16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VOXTRAL_DIR=/home/param/models/aloe-ft-vox17127-nopersona/assets/voxtral-realtime
export S4_VOXTRAL_PATH="$VOXTRAL_DIR"
export TRITON_CACHE_DIR="$NEXT_OPD_ROOT/state/triton" TORCHINDUCTOR_CACHE_DIR="$NEXT_OPD_ROOT/state/inductor"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1
export S4_FFMPEG=/home/param/pose-repo/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
