#!/bin/bash
# usage: [NPROC=6] [CUDA_VISIBLE_DEVICES=...] [VAR=override ...] ./launch_rosie_full.sh
# single host, no Slurm. Logs: /workspace/rosie-ft/runs/$ROSIE_RUN/{train.log,metrics.jsonl,cfg.json,export-*}
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/run_rosie_full.env"
NPROC=${NPROC:-6}
mkdir -p "$ROSIE_RUN_ROOT/$ROSIE_RUN"
echo "run $ROSIE_RUN on $NPROC GPUs (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all})"
exec /workspace/aloe-live/venv/bin/torchrun --standalone --nproc_per_node="$NPROC" "$HERE/train_rosie_full.py" 2>&1 | tee -a "$ROSIE_RUN_ROOT/$ROSIE_RUN/train.log"
