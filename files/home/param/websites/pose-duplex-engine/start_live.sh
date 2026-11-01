#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export FA3_DIR=/home/param/pose-repo/.cache/kernels/vllm-flash-attn3
export MIMI_DIR=/home/param/models/mimi
export VOXTRAL_DIR=/home/param/models/aloe-ft-vox17127-nopersona/assets/voxtral-realtime
export PPMLIVE_STATE_DIR=/home/param/runs/pose-runs/pose-sites-20261030/duplex-live
export PPMLIVE_PACKAGED_OPENINGS=/home/param/models/imported-weights/ppm-aloe-poseduplex-raw-final/code/voice_clips/openings-raw
export PPMLIVE_VIDEO_PEOPLE=/home/param/datasets/pose-demo/prompts
export PPMLIVE_BIND=127.0.0.1
export PPMLIVE_FEED_URL=off
export PPMLIVE_LAST_STEP=424
export PPMLIVE_DEFAULTS=nopersona
export TRITON_CACHE_DIR=/home/param/runs/pose-runs/pose-sites-20261030/triton-duplex-live
export PYTHONUNBUFFERED=1
export C_INCLUDE_PATH=/usr/include/python3.12
export CPLUS_INCLUDE_PATH=/usr/include/python3.12
cd /home/param/websites/pose-duplex-engine
exec /home/param/pose-repo/.venv/bin/python server.py --ck /home/param/models/imported-weights/ppm-aloe-poseduplex-raw-final/export-424-dcp --port 9216 --token-file "$PPMLIVE_STATE_DIR/backend.token"
