#!/usr/bin/env python3
"""One-shot SAVE_NOW request for the already-running first-band finetune."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import time

RUN = Path('/home/param/runs/pose-runs/pose-duplex-first-band-20261030')
CONTROL = RUN / 'control'
LAUNCH_ID = '20261031T041222Z-23804ec9'
JOB = RUN / 'cluster' / LAUNCH_ID
TARGET = 226
STATE = CONTROL / 'save_at_226.json'
CHECKPOINTS = Path('/home/param/models/trained-weights/pose-duplex-first-band-20261030/checkpoints/pose-duplex-first-band-20261030')
TOKEN = LAUNCH_ID + ':save-at-226'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': fields[19], 'state': fields[0]}


def ensure_launch():
    assert read(RUN / 'cluster/latest/launch.json')['launch_id'] == LAUNCH_ID, 'Launch changed'
    assert not (JOB / 'finished.json').exists(), 'Training has finished'
    saved = read(JOB / 'torchrun.json')
    live = identity(saved['pid'])
    assert live['start_ticks'] == saved['start_ticks'] and live['state'] != 'Z', 'Training process exited'


def write_state(status, **extra):
    state = {'status': status, 'launch_id': LAUNCH_ID, 'target_step': TARGET,
             'updated_utc': now(), 'watcher': identity(os.getpid()), **extra}
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(STATE)
    print(json.dumps(state), flush=True)


def completed_request():
    for marker in CONTROL.glob('SAVE_NOW.done-*'):
        try:
            if read(marker).get('request_id') != TOKEN:
                continue
        except (ValueError, OSError):
            continue
        step = int(marker.name.rsplit('-', 1)[1])
        checkpoint = CHECKPOINTS / f'dcp-{step}'
        assert (checkpoint / 'COMPLETE').is_file(), f'Save skipped or incomplete: {checkpoint}'
        state = read(checkpoint / 'train_state.json')
        assert state['step'] == step and state['world'] == 40
        assert state['hosts'] == ['landsend-2', 'landsend-3', 'landsend-4', 'landsend-5', 'landsend-12']
        assert f'[resume-save] WROTE {checkpoint} at step {step};' in (JOB / 'node.log').read_text()
        # SAVE_NOW is consumed only after save_local's final global barrier.
        write_state('complete', saved_step=step, checkpoint=str(checkpoint),
                    distributed_save_completed=True, training_continues=True)
        return True
    return False


def main():
    assert socket.gethostname().split('.')[0] == 'landsend-2'
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / 'save_at_226.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if completed_request():
            return
        ensure_launch()
        last_step = None
        while True:
            ensure_launch()
            progress = read(RUN / 'progress.json')
            step = int(progress['step'])
            if step >= TARGET:
                assert step < progress['steps'], 'Run is already at its final step'
                marker = CONTROL / 'SAVE_NOW'
                payload = {'request_id': TOKEN, 'target_step': TARGET,
                           'observed_step': step, 'requested_utc': now()}
                if marker.exists():
                    assert read(marker).get('request_id') == TOKEN, 'Another save request is pending'
                else:
                    with marker.open('x') as stream:
                        json.dump(payload, stream)
                write_state('requested', observed_step=step, control_file=str(marker))
                break
            if step != last_step:
                write_state('waiting', observed_step=step)
                last_step = step
            time.sleep(1)
        while not completed_request():
            ensure_launch()
            time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_state('failed', error=f'{type(exc).__name__}: {exc}')
        raise
