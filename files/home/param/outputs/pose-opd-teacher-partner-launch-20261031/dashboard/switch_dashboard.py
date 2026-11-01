#!/usr/bin/env python3
"""Explicit future activation/rollback; default only reports prepared settings."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
RUN = Path('/home/param/runs/pose-runs/pose-opd-teacher-partner-b336-general-20261031')
SITE = Path('/home/param/websites/pose-opd-credit-dashboard')
SETTINGS = SITE / 'pose-preview/settings.json'
UNITS = Path('/home/param/.config/systemd/user')
BACKUP = HERE / 'activation-backup'
SERVICES = ['pose-opd-credit-collector.service', 'pose-opd-credit-samples.service']
TARGETS = [(SETTINGS, HERE / 'settings.prepared.json')]
for name in ('collector', 'samples'):
    TARGETS.append((UNITS / f'pose-opd-credit-{name}.service.d/60-general-parent.conf', HERE / f'{name}.override.conf'))

def systemctl(*args):
    subprocess.run(['systemctl', '--user', *args], check=True)

def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.prepared-' + str(os.getpid()))
    temporary.write_bytes(data)
    os.replace(temporary, path)

def restore():
    records = json.loads((BACKUP / 'files.json').read_text())
    systemctl('stop', *SERVICES)
    for row in records:
        path = Path(row['path'])
        if row['existed']:
            atomic(path, (BACKUP / row['backup']).read_bytes())
        else:
            path.unlink(missing_ok=True)
    systemctl('daemon-reload')
    systemctl('start', *SERVICES)
    systemctl('restart', 'pose-opd-credit-dashboard.service')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('action', nargs='?', choices=('check', 'activate', 'rollback'), default='check')
args = parser.parse_args()
if args.action == 'rollback':
    restore()
    print('Restored the dashboard settings and service overrides captured before activation.')
elif args.action == 'check':
    print(json.dumps({'action': 'check_only', 'prepared_settings': json.loads((HERE/'settings.prepared.json').read_text()),
        'preparation_exists': (RUN/'preparation.json').is_file(),
        'trainer_has_initialized': (RUN/'config.json').is_file() and (RUN/'plan.json').is_file(),
        'activation_performed': False}, indent=2))
else:
    # Wait until the trainer has initialized this exact production run.
    config = json.loads((RUN/'config.json').read_text())
    plan = json.loads((RUN/'plan.json').read_text())
    preparation = json.loads((RUN/'preparation.json').read_text())
    assert config['method'] == preparation['identity']['method'], 'Training recipe differs from preparation'
    assert plan['digest'] == preparation['plan']['digest'], 'Training plan differs from preparation'
    assert plan['world'] == 56 and plan['effective_batch'] == 336
    assert not BACKUP.exists(), 'Activation backup exists; inspect it before activating again'
    BACKUP.mkdir()
    records = []
    for number, (path, _) in enumerate(TARGETS):
        if path.is_symlink():
            raise RuntimeError(f'Refusing to replace a symlink: {path}')
        record = {'path': str(path), 'existed': path.exists(), 'backup': f'file-{number}'}
        if path.exists():
            shutil.copy2(path, BACKUP/record['backup'])
        records.append(record)
    (BACKUP/'files.json').write_text(json.dumps(records, indent=2)+'\n')
    settings = json.loads((HERE/'settings.prepared.json').read_text())
    samples = Path(settings['samples_dir']); samples.mkdir(parents=True, exist_ok=True)
    manifest = samples/'manifest.json'
    if not manifest.exists():
        atomic(manifest, b'{"schema_version":1,"synthetic":false,"rollouts":[]}\n')
    try:
        systemctl('stop', *SERVICES)
        for path, source in TARGETS:
            atomic(path, source.read_bytes())
        systemctl('daemon-reload')
        systemctl('start', *SERVICES)
        systemctl('restart', 'pose-opd-credit-dashboard.service')
    except BaseException:
        restore()
        raise
    print('Activated the prepared new-run telemetry and decoder; website layout and tunnel unchanged.')
