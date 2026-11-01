#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/home/param/pose-runs/opd-next-web-20261030
TRAIN_ROOT=/home/param/pose-runs/opd-next-rosie-conditioned-20261030/training
echo "Waiting for committed audio-conditioned epoch-2 serving export and approved Rosie reference."
until [[ -f "$TRAIN_ROOT/TRAINING_COMPLETE.json" && -f "$RUN_ROOT/export-conditioned-epoch-2/COMPLETE" && -f "$RUN_ROOT/conditioning/REFERENCE_APPROVED.json" ]]; do
  sleep 2
done
echo "Serving export complete and reference approval present; starting two-GPU backend."
exec /bin/bash "$RUN_ROOT/tools/backend.sh"
