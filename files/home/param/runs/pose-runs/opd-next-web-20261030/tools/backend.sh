#!/usr/bin/env bash
set -euo pipefail
[[ $(hostname -s) == landsend-5 ]]
export CUDA_VISIBLE_DEVICES=0,1
python3 - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path
assert Path('/home/param/pose-runs/opd-next-rosie-conditioned-20261030/training/TRAINING_COMPLETE.json').is_file()
conditioning_root = Path('/home/param/pose-runs/opd-next-web-20261030/conditioning')
approval = json.loads((conditioning_root / 'REFERENCE_APPROVED.json').read_text())
config = json.loads((conditioning_root / 'conditioning.json').read_text())
codes_path = Path(config['codes_path'])
if not codes_path.is_absolute():
    codes_path = conditioning_root / codes_path
codes_sha256 = hashlib.sha256(codes_path.read_bytes()).hexdigest()
assert approval.get('approved') is True, 'Rosie reference requires explicit user approval'
assert approval.get('codes_sha256') == config.get('codes_sha256') == codes_sha256, 'Approved Rosie reference does not match configured audio codes'
rows = subprocess.check_output(['nvidia-smi', '-i', '0,1', '--query-gpu=index,memory.used',
                                '--format=csv,noheader,nounits'], text=True)
memory = {int(row.split(',')[0]): int(row.split(',')[1]) for row in rows.splitlines()}
assert all(value == 0 for value in memory.values()), f'Inference GPUs must be free after training exits: {memory}'
PY
source /home/param/pose-runs/opd-next-web-20261030/tools/environment.sh
export PPMLIVE_SWAP_TO="$OPD_WEB_ROOT/export-conditioned-epoch-2" PPMLIVE_DEFAULTS=unbiased PPMLIVE_LAST_STEP=54
export ROSIE_CONDITIONING_CONFIG="$OPD_WEB_ROOT/conditioning/conditioning.json"
export PPMLIVE_STATE_DIR="$OPD_WEB_ROOT/state" PPMLIVE_BIND=127.0.0.1
mkdir -p "$PPMLIVE_STATE_DIR/sessions"
[[ -f "$OPD_WEB_ROOT/export-conditioned-epoch-2/COMPLETE" ]]
[[ -f "$ROSIE_CONDITIONING_CONFIG" ]]
exec /home/param/pose-repo/.venv/bin/python -u "$OPD_WEB_ROOT/srv/server.py" \
  --ck "$OPD_WEB_ROOT/export-conditioned-epoch-2" --port 8918 --token-file "$OPD_WEB_ROOT/backend.token" \
  --sessions "$OPD_WEB_ROOT/state/sessions"
