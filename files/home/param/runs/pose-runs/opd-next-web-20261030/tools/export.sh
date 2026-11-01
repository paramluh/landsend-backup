#!/bin/bash
set -euo pipefail
[ "$(hostname -s)" = landsend-5 ]
source /home/param/pose-runs/next-opd-demo-20261030/tools/environment.sh
SNAP="$NEXT_OPD_ROOT/checkpoints/checkpoint-1793328114351120738-d62d13df4f14"
[ -f "$SNAP/SNAPSHOT_COMPLETE" ]
echo "Expected export 5-10 minutes: full1439 trained tensors, BF16 weights with FP32 gain; CPU only. $(date -u +%FT%TZ)"
exec /home/param/pose-repo/.venv/bin/python -u "$NEXT_OPD_ROOT/tools/export_inference.py" --full-model --model-dcp "$SNAP/model" --manifest "$SNAP/MANIFEST.json" --base "$NEXT_OPD_ROOT/base-config" --output "$NEXT_OPD_ROOT/export-v95" --student-version 95 --shards 16
