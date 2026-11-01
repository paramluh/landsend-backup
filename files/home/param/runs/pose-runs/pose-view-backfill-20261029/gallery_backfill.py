import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/home/param/pose-repo/infer/transcript_pose')
from pose_views import build_pose_views

AUDIT = Path('/home/param/pose-runs/pose-view-backfill-20261029')
PUBLISH = '/home/param/pose-demo-website/comparison/publish.py'
ROOTS = [Path('/home/param/pose-runs/controlled-comparison-20261028'),
         Path('/home/param/pose-runs/native-comparison-20261028')]
jobs = []
for root in ROOTS:
    manifest = json.loads((root / 'gallery/manifest.json').read_text())
    (AUDIT / f'{root.name}-manifest-before.json').write_text(json.dumps(manifest, indent=2) + '\n')
    selection = {c['id']: c for c in json.loads((root / 'selection.json').read_text())['clips']}
    for clip in manifest['clips']:
        for recipe, output in clip['outputs'].items():
            assert output['status'] == 'complete'
            directory = root / 'outputs' / selection[clip['id']]['gallery_id'] / recipe
            jobs.append((root, clip['id'], recipe, directory, output['sha256']))
assert len(jobs) == 40


def run(job):
    root, clip, recipe, directory, source_sha = job
    start = time.monotonic()
    assert hashlib.sha256((directory / 'video.mp4').read_bytes()).hexdigest() == source_sha
    assert (directory / 'pose/skeleton.mp4').is_file()
    receipt = build_pose_views(directory / 'video.mp4', directory / 'pose', directory)
    assert receipt['source_video']['sha256'] == source_sha
    command = [sys.executable, PUBLISH, '--data-root', str(root / 'gallery'), 'publish',
               '--clip-id', clip, '--model', recipe, '--video', str(directory / 'video.mp4'),
               '--pose-video', str(directory / 'pose.mp4'), '--pose-overlay-video', str(directory / 'overlay.mp4'),
               '--metadata', str(directory / 'public-metadata.json')]
    subprocess.run(command, capture_output=True, text=True, check=True)
    row = dict(status='pass', root=str(root), clip=clip, recipe=recipe, source_sha256=source_sha,
               frames=receipt['frames'], seconds=receipt['seconds'], cpu_only=receipt['cpu_only'],
               source_pose_kind=receipt['source_pose']['kind'],
               pose_sha256=receipt['artifacts']['pose.mp4']['sha256'],
               pose_overlay_sha256=receipt['artifacts']['overlay.mp4']['sha256'],
               audio_packet_sha256=receipt['audio_packet_sha256'],
               audio_pcm_s16le_sha256=receipt['audio_pcm_s16le_sha256'],
               elapsed_seconds=time.monotonic()-start)
    print(json.dumps(row), flush=True)
    return row


results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for row in pool.map(run, jobs):
        results.append(row)
        (AUDIT / 'gallery-progress.json').write_text(json.dumps(results, indent=2) + '\n')
assert len(results) == 40
(AUDIT / 'gallery-backfill.json').write_text(json.dumps(dict(status='pass', cells=40, results=results), indent=2)+'\n')
print('ALL_FORTY_GALLERY_VIEWS_READY', flush=True)
