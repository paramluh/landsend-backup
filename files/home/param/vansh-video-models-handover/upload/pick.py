"""
The refs sink_universal.py gave each rollout clip: the same AloeChunkSource(...).load_eval_refs(n, like) call on the
rollout yaml (csv, root_dir, refs.bank, pose_file, rows), `like` = the chunk's first video latent as float32. The rng
is replayed to name the picked file / image per slot, and every picked slot is checked against the bank image.
Needs the dataset and the bank, so run it where the rollout ran (CPU). The yaml's relative paths resolve against
production/, as in sink_universal.py.
  python pick.py YAML OUT_DIR
-> OUT_DIR/refs_<n>.safetensors {latents [C, 9, h, w], present [9]}, OUT_DIR/manifest.json
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

PRODUCTION = Path(__file__).resolve().parents[1] / 'production'
sys.path.insert(0, str(PRODUCTION.parent))
from dataloading.ds_aloe import EVAL_REF_SEED, AloeChunkSource  # noqa: E402
from dataloading.sources import NUM_REFS  # noqa: E402
from inference.common import resolve_path  # noqa: E402

config_path, out = sys.argv[1], sys.argv[2]
config = yaml.safe_load(open(config_path))
csv_path = resolve_path(config['csv'], PRODUCTION)
rows = list(csv.DictReader(open(csv_path)))
root_dir = resolve_path(config['root_dir'], PRODUCTION)
source = AloeChunkSource(root_dir, os.path.dirname(csv_path), os.path.basename(csv_path),
                         bank=resolve_path(config['refs']['bank'], PRODUCTION), pose_file=config['pose_file'],
                         mimi_file='mimi.safetensors')  # refs only: the codes are never read
os.makedirs(out, exist_ok=True)
manifest = {}
for n in config['rows']:
    with safe_open(os.path.join(root_dir, rows[n]['paths'], 'video_latents.safetensors'), framework='pt') as handle:
        like = handle.get_slice('latents')[0:1].permute(1, 0, 2, 3).float()
    video, present = source.load_eval_refs(n, like)
    rng = np.random.default_rng([EVAL_REF_SEED, int(source.chunk_ids[n])])
    uuid = source.clips[n].uuid
    picks = []
    for slot in range(NUM_REFS):
        path = os.path.join(source.bank, uuid, f'property_{slot + 1}', 'latents.safetensors')
        if not os.path.exists(path):
            picks.append(None)
            continue
        with safe_open(path, framework='pt') as handle:
            images = handle.get_slice('video')
            count = images.get_shape()[0]
            index = int(rng.integers(count))
            same = torch.equal(video[:, slot], images[index:index + 1][0].float())
        picks.append({'file': path, 'index': index, 'of': count, 'same': same})
    save_file({'latents': video.contiguous(), 'present': present.to(torch.uint8)}, f'{out}/refs_{n}.safetensors')
    manifest[n] = {'uuid': uuid, 'paths': rows[n]['paths'], 'present': present.int().tolist(), 'picks': picks}
    print(n, uuid, present.int().tolist(), all(p['same'] for p in picks if p), flush=True)
json.dump(manifest, open(f'{out}/manifest.json', 'w'), indent=1)
