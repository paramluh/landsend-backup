#!/bin/bash
set -euo pipefail
[ "$(hostname -s)" = landsend-5 ]
export CUDA_VISIBLE_DEVICES=6,5
python3 - <<'PY'
import subprocess
readings = subprocess.check_output([
    'nvidia-smi', '-i', '6,5', '--query-gpu=index,memory.used',
    '--format=csv,noheader,nounits'], text=True)
memory = {int(row.split(',')[0]): int(row.split(',')[1]) for row in readings.splitlines()}
assert memory[6] == 0 and memory[5] <= 12000, \
    f'Insufficient free memory on inference pair 6,5: {memory}'
PY
source /home/param/pose-runs/next-opd-demo-20261030/tools/environment.sh
export PPMLIVE_SWAP_TO="$NEXT_OPD_ROOT/export-v95" PPMLIVE_DEFAULTS=unbiased PPMLIVE_LAST_STEP=95 PPMLIVE_STATE_DIR="$NEXT_OPD_ROOT/state" PPMLIVE_BIND=127.0.0.1
mkdir -p "$PPMLIVE_STATE_DIR/sessions"
[ -f "$NEXT_OPD_ROOT/export-v95/COMPLETE" ]
echo "Expected model load+capture+smoke 2-5 minutes; streaming target <80ms/frame. $(date -u +%FT%TZ)"
exec /home/param/pose-repo/.venv/bin/python -u "$NEXT_OPD_ROOT/srv/server.py" --ck "$NEXT_OPD_ROOT/export-v95" --port 8931 --token-file "$NEXT_OPD_ROOT/backend.token" --sessions "$NEXT_OPD_ROOT/state/sessions"
