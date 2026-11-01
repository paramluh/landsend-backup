"""Gracefully freeze the active run and verify its HF backup before rental expiry."""
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

SOURCE = Path('/home/param/pose-worktrees/pose-basic-opd')
sys.path.insert(0, str(SOURCE))
from finetune.pose_opd_credit.checkpoint_policy import (
    atomic_json, RequiredCheckpoint, _completed_deadline_checkpoint,
)

ROOT = Path('/home/param/runs/pose-runs/pose-basic-opd-b336-20261031')
OUT = Path('/home/param/outputs/landsend-backup-20261031')
STOP_AT = '2026-11-01T06:15:00+00:00'
EXPIRES_AT = '2026-11-01T07:00:00+00:00'
ORIGINAL_REQUIRED_AT = '2026-11-01T06:50:00+00:00'
EFFECTIVE_REQUIRED_AT = '2026-11-01T06:30:00+00:00'


def status(state, **extra):
    atomic_json(OUT / 'final-training-backup.json', {
        'state': state, 'time': time.time(), 'pid': os.getpid(),
        'stop_at_utc': STOP_AT, 'rental_expires_at_utc': EXPIRES_AT,
        'run_dir': str(ROOT), **extra,
    })


def main():
    deadline = dt.datetime.fromisoformat(STOP_AT).timestamp()
    status('armed')
    result_file = ROOT / 'RESULT.json'
    while time.time() < deadline and not result_file.exists():
        time.sleep(min(20, deadline - time.time()))
    if not result_file.exists():
        (ROOT / 'control').mkdir(exist_ok=True)
        (ROOT / 'control/STOP_AFTER_SAVE').touch()
        (ROOT / 'control/SAVE_NOW').touch()
        status('graceful-save-and-stop-requested')
    while not result_file.exists():
        time.sleep(10)
    result = json.loads(result_file.read_text())
    assert result['checkpointing_enabled'] and not result['synthetic_shape_only']
    completed = _completed_deadline_checkpoint(ROOT, 0)
    assert completed is not None, 'Final checkpoint failed local completion validation'
    checkpoint = completed['checkpoint']
    assert checkpoint['step'] == result['next_update'], 'Final result does not match completed checkpoint'
    assert Path(result['last_checkpoint']).resolve() == (Path(checkpoint['path']) / 'resume').resolve()
    # A gracefully stopped run is now static, even if its three epochs are incomplete.
    # This lets the existing 23:30 watcher recognize the verified final checkpoint.
    policy = RequiredCheckpoint(ROOT, ORIGINAL_REQUIRED_AT)
    policy.record(checkpoint, final=True)
    receipt = json.loads(policy.receipt.read_text())
    receipt.update(effective_required_at_utc=EFFECTIVE_REQUIRED_AT,
                   rental_final_stop_at_utc=STOP_AT,
                   reason='rental-expiry-graceful-stop-model-is-static')
    atomic_json(policy.receipt, receipt)
    status('stopped-waiting-for-hf-verification', step=checkpoint['step'],
           total_updates=result['total_updates'], checkpoint=checkpoint)
    publication = ROOT / 'hf-upload' / Path(checkpoint['path']).name / 'landsend-0/published.json'
    while not publication.exists():
        time.sleep(15)
    published = json.loads(publication.read_text())
    latest = published['latest']
    assert latest['step'] == checkpoint['step'], 'HF latest does not identify final checkpoint'
    assert latest['world'] == 56 and len(latest['hosts']) == 7
    assert latest['verified_files'], 'Missing all-host remote verification'
    status('verified', step=checkpoint['step'], total_updates=result['total_updates'],
           checkpoint=checkpoint, hf_repo='InternalCan/pose-basic-opd-b336-20261031',
           revision=published['revision'], verified_hosts=latest['hosts'],
           verified_file_count=len(latest['verified_files']))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('error', error=f'{type(exc).__name__}: {exc}')
        raise
