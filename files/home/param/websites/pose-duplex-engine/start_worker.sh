#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2,3
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export FA3_DIR=/home/param/pose-repo/.cache/kernels/vllm-flash-attn3
export MIMI_DIR=/home/param/models/mimi
export VOXTRAL_DIR=/home/param/models/aloe-ft-vox17127-nopersona/assets/voxtral-realtime
export POSE_DUPLEX_ENGINE=/home/param/websites/pose-duplex-engine
export GENWEB_CONVS=/home/param/datasets/pose-duplex/convs
export GENWEB_PEOPLE=/home/param/datasets/pose-demo/prompts
export GENWEB_ACTORS=/home/param/datasets/pose-demo/actors
export GENWEB_DUP_CK=/home/param/models/imported-weights/ppm-aloe-poseduplex-raw-final/export-424-dcp
export PPMLIVE_DEFAULTS=nopersona
export TRITON_CACHE_DIR=/home/param/runs/pose-runs/pose-sites-20261030/triton-duplex-worker
export PYTHONUNBUFFERED=1
export C_INCLUDE_PATH=/usr/include/python3.12
export CPLUS_INCLUDE_PATH=/usr/include/python3.12
cd /home/param/websites/pose-playground
exec /home/param/pose-repo/.venv/bin/python worker_dup.py duplexlandsend12
