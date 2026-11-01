#!/usr/bin/env python3
"""Read-only public/media verification after the existing 120-second monitor."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent
MIRROR = Path('/home/param/outputs/pose-opd-symmetric-b336-website-pad0-live-20261031')
RUN_ID = 'pose-opd-symmetric-b336-website-pad0-20261031'
BASE = 'https://mph-daniel-containers-afternoon.trycloudflare.com/'
monitor = json.loads((HERE.parent / 'FIRST-TWO-VERIFIED.json').read_text())
assert max(row['update'] for row in monitor['metrics']) >= 2

def public(name):
    with urllib.request.urlopen(BASE + name + '?v=' + str(time.time_ns()), timeout=20) as response:
        assert response.status == 200
        return json.load(response)

progress = public('opd_prog.json')
chart = public('s3live_pose_opd_credit.json')
samples = public('samples/manifest.json')
assert progress['root'] == chart['run'] == RUN_ID
assert progress['completed'] >= 2 and chart['step'] >= 2
assert progress['collector']['status'] == 'live'
assert {branch['step'] for branch in chart['branches']} == {89, 13}
raw = []
jobs = []
for update in (1, 2):
    manifest_path = MIRROR / 'raw-rollouts' / f'update-{update:06d}' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['update'] == update and len(manifest['samples']) == 4
    source_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    job_id = f'update-{update:06d}-{source_sha[:16]}'
    job = json.loads((MIRROR / 'video-bridge/jobs' / job_id / 'job.json').read_text())
    submitted = json.loads((MIRROR / 'video-bridge/submitted' / (job_id + '.json')).read_text())
    assert job['run_id'] == RUN_ID and job['source_manifest_sha256'] == source_sha
    assert submitted['submitted'] == job_id and len(job['samples']) == 4
    for sample in manifest['samples']:
        assert sample['conversation_mode'] == 'symmetric_streaming'
        assert sample['incoming_audio_delay_samples'] == sample['output_audio_delay_samples'] == 3912
        file = manifest_path.parent / sample['file']
        assert file.stat().st_size == sample['bytes']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == sample['sha256']
    raw.append({'update': update, 'sample_count': 4, 'source_manifest_sha256': source_sha,
                'files_verified': True, 'job_id': job_id, 'code_bytes': submitted['code_bytes']})
    jobs.append(job_id)

decoded = []
for rollout in samples['rollouts']:
    if rollout['update'] not in (1, 2):
        continue
    assert len(rollout['samples']) == 4
    for sample in rollout['samples']:
        assert sample['pose_offset_seconds'] == .163
        assert sample['output_audio_source'] == 'partner_received_pcm'
        assert sample['conversation_mode'] == 'symmetric_streaming'
        if sample.get('video_url'):
            assert sample['video_offset_seconds'] == .163
    decoded.append({'update': rollout['update'], 'sample_count': 4,
                    'video_count': sum(bool(s.get('video_url')) for s in rollout['samples']),
                    'pose_video_offset_seconds': .163, 'audio_source': 'partner_received_pcm'})
assert any(row['update'] == 1 for row in decoded)
bridge = json.loads((MIRROR / 'video-bridge/health.json').read_text())
assert bridge['status'] == 'running' and bridge['errors'] == 0

remote_code = '''import json,pathlib,sys,subprocess
root=pathlib.Path('/home/param/outputs/pose-opd-video-worker-20261031');ids=json.loads(sys.argv[1]);jobs=[]
for name in ids:
 p=root/'incoming'/name;job=json.loads((p/'job.json').read_text());progress=p/'progress.json'
 jobs.append({'job_id':name,'run_id':job['run_id'],'update':job['update'],'batch':len(job['samples']),
   'progress':json.loads(progress.read_text()) if progress.exists() else None,
   'completed_samples':len([f for f in (root/'completed'/name).glob('*.json') if f.name!='result.json'])})
print(json.dumps({'status':json.loads((root/'status.json').read_text()),'jobs':jobs,
  'service':subprocess.check_output(['systemctl','--user','show','pose-opd-video-worker.service','-p','MainPID','-p','NRestarts','-p','ActiveState'],text=True)}))
'''
remote = json.loads(subprocess.check_output(['ssh', '-A', '-i', '/home/param/.ssh/id_ed25519',
    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'param@89.169.125.72',
    shlex.join(['python3', '-c', remote_code, json.dumps(jobs)])], text=True, timeout=20))
assert all(job['run_id'] == RUN_ID and job['batch'] == 4 for job in remote['jobs'])
assert any(job['progress'] is not None for job in remote['jobs'])
assert 'ActiveState=active' in remote['service']
record = {'verified_at': time.time(), 'run': RUN_ID, 'monitor_receipt': str(HERE.parent / 'FIRST-TWO-VERIFIED.json'),
    'public_progress': {k: progress[k] for k in ['completed', 'update', 'max', 'state']},
    'public_chart': {'step': chart['step'], 'max_steps': chart['max_steps'],
                     'references': [{'run': b['run'], 'step': b['step']} for b in chart['branches']]},
    'raw': raw, 'decoded_public': decoded, 'bridge': bridge, 'work_vm': remote,
    'training_polled': False}
(HERE / 'FIRST-TWO-PIPELINE-VERIFIED.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
