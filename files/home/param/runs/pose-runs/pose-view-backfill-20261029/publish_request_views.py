"""Publish completed CPU pose views without modifying generation or source video."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

assert socket.gethostname().split('.')[0] == 'landsend-1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/home/param/pose-repo/infer/transcript_pose')
from pose_views import validate_pose_views

QUEUE = Path('/home/param/pose-runs/inference-requests-20261028/jobs')
OUT = Path('/home/param/pose-runs/pose-view-backfill-20261029')
OUT.mkdir(exist_ok=True)


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    pending = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    pending.write_text(json.dumps(data, indent=2) + '\n')
    pending.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


done, waiting = [], []
for job in sorted(QUEUE.iterdir()):
    if not (job / 'state.json').is_file() or read(job / 'state.json').get('status') != 'complete':
        continue
    source = job / 'result' if (job / 'result/metadata.json').is_file() else job
    if not (source / 'pose-views.json').is_file():
        waiting.append(job.name)
        continue
    receipt = validate_pose_views(source, video=source / 'video.mp4')
    meta = read(source / 'metadata.json')
    assert sha(source / 'video.mp4') == sha(job / 'video.mp4') == meta['sha256'] == receipt['source_video']['sha256']
    fields = dict(pose_video='pose.mp4', pose_sha256=receipt['artifacts']['pose.mp4']['sha256'],
                  pose_overlay_video='overlay.mp4', pose_overlay_sha256=receipt['artifacts']['overlay.mp4']['sha256'])
    meta.update(**fields, pose_views=receipt)
    write(source / 'metadata.json', meta)
    if source != job:
        for name in ('pose.mp4', 'overlay.mp4', 'pose-views.json', 'metadata.json'):
            temp = job / f'.pose-publish-{os.getpid()}-{name}'
            temp.unlink(missing_ok=True)
            os.link(source / name, temp)
            temp.replace(job / name)
    state = read(job / 'state.json')
    assert state['status'] == 'complete'
    state.update(**fields, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(job / 'state.json', state)
    assert sha(job / 'video.mp4') == meta['sha256']
    done.append(dict(id=job.name, status='pass', frames=receipt['frames'], seconds=receipt['seconds'],
                     original_sha256=meta['sha256'], **fields))
    print(json.dumps(done[-1]), flush=True)

report = dict(status='pass' if not waiting else 'partial', completed=len(done), waiting=waiting, jobs=done)
write(OUT / 'request-publication-audit.json', report)
print(json.dumps(dict(status=report['status'], completed=len(done), waiting=waiting)), flush=True)
