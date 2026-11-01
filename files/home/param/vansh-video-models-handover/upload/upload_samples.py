"""
label.py OUT_DIR -> dataset vanshjjw/aloe-samples under samples/<RUN>/step<STEP>-<WEIGHTS>-<CLIPSET>/: stages
OUT_DIR/hf/ (each mp4 remuxed faststart, metadata.csv in the dataset's schema from manifest.csv, plus refs_present /
refs_in_chunk when label.py wrote them, the rollout YAML(s)), then one commit that adds them and deletes every other
remote file under the prefix, then checks every file's size and sha256 (LFS) or git blob sha1 on the remote.
Token from $HF_TOKEN.
  HF_TOKEN=... python upload_samples.py OUT_DIR RUN STEP WEIGHTS CLIPSET YAML [YAML ...]
"""
import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

REPO = 'vanshjjw/aloe-samples'
COLS = ['file_name', 'run', 'step', 'weights', 'clipset', 'clip', 'clip_id', 'actor', 'kind', 'category', 'speaker',
        'duration', 'seed', 'train_hours', 'recording_id']
src, RUN, STEP, WEIGHTS, CLIPSET, yamls = Path(sys.argv[1]), *sys.argv[2:6], [Path(p) for p in sys.argv[6:]]
prefix = f'samples/{RUN}/step{STEP}-{WEIGHTS}-{CLIPSET}'
stage = src / 'hf'
stage.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(src / 'manifest.csv')))
extra = [k for k in ('refs_present', 'refs_in_chunk') if k in rows[0]]
with open(stage / 'metadata.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(COLS + extra)
    for r in rows:
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(src / r['file']), '-map', '0', '-c', 'copy',
                        '-movflags', '+faststart', str(stage / r['file'])], check=True)
        w.writerow([r['file'], RUN, STEP, WEIGHTS, CLIPSET, r['index'], r['uuid'], r['actor'], r['kind'], r['category'],
                    r['speaker'], r['duration'], r['seed'], f"{float(r['train_hours']):.2f}", r['recording_id'],
                    *[r[k] for k in extra]])
for yaml_path in yamls:
    shutil.copy2(yaml_path, stage / yaml_path.name)

api = HfApi()
files = sorted(stage.iterdir())
names = {f'{prefix}/{p.name}' for p in files}
remote = [p for p in api.list_repo_files(REPO, repo_type='dataset') if p.startswith(prefix + '/')]
ops = [CommitOperationDelete(p) for p in remote if p not in names]
ops += [CommitOperationAdd(f'{prefix}/{p.name}', str(p)) for p in files]
api.create_commit(REPO, ops, repo_type='dataset', commit_message=f'{prefix}: {len(files)} files')

info = {f.path: f for f in api.list_repo_tree(REPO, path_in_repo=prefix, repo_type='dataset') if hasattr(f, 'size')}
bad = []
for p in files:
    r, data = info.get(f'{prefix}/{p.name}'), p.read_bytes()
    ok = r is not None and r.size == len(data) and (
        r.lfs.sha256 == hashlib.sha256(data).hexdigest() if r.lfs else
        r.blob_id == hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest())
    bad += [] if ok else [p.name]
print(f'{prefix}: deleted {len(ops) - len(files)}, uploaded {len(files)}, verified {len(files) - len(bad)}, bad {bad}, '
      f'remote == local: {set(info) == names}', flush=True)
