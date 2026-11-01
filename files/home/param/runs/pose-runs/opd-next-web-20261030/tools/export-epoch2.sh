#!/usr/bin/env bash
set -euo pipefail
[[ $(hostname -s) == landsend-5 ]]
source /home/param/pose-runs/opd-next-web-20261030/tools/environment.sh
export CUDA_VISIBLE_DEVICES=""
SFT_RUN=/home/param/pose-runs/opd-next-rosie-conditioned-20261030/training
[[ -f "$SFT_RUN/TRAINING_COMPLETE.json" ]]
[[ -f "$SFT_RUN/epoch-2/COMPLETE" ]]
exec /home/param/pose-repo/.venv/bin/python -u "$OPD_WEB_ROOT/tools/export_inference.py" \
  --full-model --model-dcp "$SFT_RUN/epoch-2/model" \
  --manifest "$SFT_RUN/epoch-2/MANIFEST.json" --base "$OPD_WEB_ROOT/base-config" \
  --output "$OPD_WEB_ROOT/export-conditioned-epoch-2" --student-version 110 --shards 16
