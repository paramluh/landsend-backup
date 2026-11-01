#!/usr/bin/env python3
"""Activate this exact symmetric production feed; preserve layout and VM worker."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RUN = Path('/home/param/runs/pose-runs/pose-opd-symmetric-b336-website-pad0-20261031')
SPEC = HERE.parent / 'snapshot-1793472444394221943/production-launch.json'
SETTINGS = Path('/home/param/websites/pose-opd-credit-dashboard/pose-preview/settings.json')
UNITS = Path('/home/param/.config/systemd/user')
BACKUP = HERE / 'activation-backup'
SERVICES = ['pose-opd-credit-collector.service', 'pose-opd-credit-samples.service', 'pose-opd-video-bridge.service']
TARGETS = [(SETTINGS, HERE / 'settings.prepared.json')]
for service, prepared in zip(SERVICES, ['collector', 'samples', 'video-bridge']):
    TARGETS.append((UNITS / (service + '.d/70-symmetric-streaming.conf'), HERE / (prepared + '.override.conf')))

def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.prepared-' + str(os.getpid()))
    tmp.write_bytes(data)
    os.replace(tmp, path)

def systemctl(*args):
    subprocess.run(['systemctl', '--user', *args], check=True)

def validate():
    spec = json.loads(SPEC.read_text())
    go = json.loads((RUN / 'supervisor/GO.json').read_text())
    recipe_path = Path(spec['config'])
    assert hashlib.sha256(recipe_path.read_bytes()).hexdigest() == spec['config_sha256'] == go['config_sha256']
    assert go['source_commit'] == spec['source_commit'] == 'd709e280adf99fc04838c40b855c5c6e4feeb42e'
    module_spec = importlib.util.spec_from_file_location('activation_training_config', RUN / 'code/finetune/pose_opd_credit/config.py')
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    expected = json.loads(json.dumps(module.TrainingConfig.from_dict(json.loads(recipe_path.read_text())).to_dict()))
    config = json.loads((RUN / 'config.json').read_text())
    plan = json.loads((RUN / 'plan.json').read_text())
    assert config['method'] == expected, 'Actual trainer method differs from immutable recipe'
    assert config['actor'] == 'frozen-teacher-partner-symmetric-streaming', config['actor']
    assert config['world'] == plan['world'] == 56
    assert plan['effective_batch'] == 336 and plan['rows_per_rank'] == 6
    assert plan['epochs'] == 3 and len(plan['updates']) == 123
    assert config['plan_digest'] == plan['digest']
    assert expected['partner_sampling'] == 'website_20261031' and expected['partner_pad_bias'] == 0
    assert expected['partner_conversation_mode'] == 'symmetric_streaming'
    assert expected['partner_audio_delay_samples'] == 0
    for name in ['benchmark_shape_only', 'benchmark_load_rollouts', 'benchmark_replay_microbatches', 'benchmark_microbatches', 'benchmark_warm_optimizer']:
        assert not config.get(name), (name, config.get(name))
    return {'source_commit': go['source_commit'], 'config_sha256': go['config_sha256'],
            'plan_digest': plan['digest'], 'world': plan['world'], 'updates': len(plan['updates']),
            'effective_batch': plan['effective_batch'], 'actor': config['actor']}

def restore():
    systemctl('stop', *SERVICES)
    for record in json.loads((BACKUP / 'files.json').read_text()):
        path = Path(record['path'])
        if record['existed']:
            atomic(path, (BACKUP / record['backup']).read_bytes())
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
    print('Restored previous feed settings and local service overrides.')
else:
    verified = validate()
    if args.action == 'check':
        print(json.dumps({'verified_runtime': verified, 'activation_performed': False}, indent=2))
    else:
        assert not BACKUP.exists(), 'Existing activation backup: inspect before reactivation'
        BACKUP.mkdir()
        records = []
        for number, (path, _) in enumerate(TARGETS):
            assert not path.is_symlink(), path
            record = {'path': str(path), 'existed': path.exists(), 'backup': f'file-{number}'}
            if path.exists():
                shutil.copy2(path, BACKUP / record['backup'])
            records.append(record)
        (BACKUP / 'files.json').write_text(json.dumps(records, indent=2) + '\n')
        settings = json.loads((HERE / 'settings.prepared.json').read_text())
        samples = Path(settings['samples_dir']); samples.mkdir(parents=True, exist_ok=True)
        manifest = samples / 'manifest.json'
        if not manifest.exists():
            atomic(manifest, b'{"schema_version":1,"synthetic":false,"rollouts":[]}\n')
        try:
            systemctl('stop', *SERVICES)
            for target, source in TARGETS:
                atomic(target, source.read_bytes())
            systemctl('daemon-reload')
            systemctl('start', *SERVICES)
            systemctl('restart', 'pose-opd-credit-dashboard.service')
        except BaseException:
            restore()
            raise
        receipt = {'activated_at': time.time(), 'verified_runtime': verified, 'settings': settings,
                   'unchanged': ['sample_layout', 'current_run_zoom_default', 'tunnel', 'work_vm_video_worker']}
        atomic(HERE / 'activation-receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())
        print(json.dumps(receipt, indent=2))
