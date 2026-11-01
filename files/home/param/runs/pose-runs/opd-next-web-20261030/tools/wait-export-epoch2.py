#!/usr/bin/env python3
"""Wait for the committed final SFT epoch, then perform one CPU-only export."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/param/pose-runs/opd-next-web-20261030')
TRAINING = Path('/home/param/pose-runs/opd-next-rosie-conditioned-20261030/training')
STATE = ROOT / 'EXPORT_CONDITIONED_STATE.json'


def write_state(status, **values):
    state = dict(status=status, watcher_pid=os.getpid(),
                 updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
    temporary = STATE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATE)


write_state('waiting_for_training_complete')
while not (TRAINING / 'TRAINING_COMPLETE.json').is_file():
    time.sleep(15)
complete = json.loads((TRAINING / 'TRAINING_COMPLETE.json').read_text())
assert complete['epochs'] == 2 and complete['steps'] == 54
assert (TRAINING / 'epoch-2/COMPLETE').is_file()
manifest = json.loads((TRAINING / 'epoch-2/MANIFEST.json').read_text())
assert manifest['checkpoint_id'] == 'rosie-conditioned-epoch-2-step-54'
with (ROOT / 'logs/export-conditioned-epoch2.log').open('xb') as log:
    child = subprocess.Popen(['/bin/bash', str(ROOT / 'tools/export-epoch2.sh')],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    write_state('exporting', exporter_pid=child.pid)
    code = child.wait()
write_state('complete' if code == 0 else 'failed', exit_code=code,
            output=str(ROOT / 'export-conditioned-epoch-2'), log=str(ROOT / 'logs/export-conditioned-epoch2.log'))
raise SystemExit(code)
