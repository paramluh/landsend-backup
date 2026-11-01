#!/usr/bin/env bash
set -euo pipefail
export CANOPY_ROOT=/home/param
export GENWEB_MONO_ENGINE=/home/param/websites/pose-monoplex-engine
export GENWEB_LIVE=/home/param/websites/pose-duplex-engine
export GENWEB_PEOPLE=/home/param/datasets/pose-demo/prompts
export GENWEB_ACTORS=/home/param/datasets/pose-demo/prompts
export GENWEB_CK=/home/param/models/imported-weights/ppm-aloe-monoplex-pose-raw-bandfix-355/weights/export-355
export GENWEB_WEIGHTS=base
export CUDA_VISIBLE_DEVICES=0,1
export MIMI_DIR=/home/param/models/mimi
export FA3_DIR=/home/param/pose-repo/.cache/kernels/vllm-flash-attn3
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 TRITON_CACHE_DIR=/home/param/runs/pose-runs/pose-sites-20261030/triton-mono
export PATH=/home/param/pose-repo/.venv/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$TRITON_CACHE_DIR" /home/param/runs/pose-runs/pose-sites-20261030/logs
cd /home/param/websites/pose-playground
exec /home/param/pose-repo/.venv/bin/python -u worker.py mono-landsend12 >> /home/param/runs/pose-runs/pose-sites-20261030/logs/monoplex.log 2>&1
