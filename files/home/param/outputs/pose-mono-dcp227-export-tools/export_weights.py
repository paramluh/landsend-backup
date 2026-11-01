"""CPU-only native inference export from immutable pose-duplex dcp-227 saves.

Only model mmap pages are read; optimizer/RNG/scheduler state is excluded.
BF16 tensors match the trainer's normal inference export. VGain remains FP32.
The caller gathers host outputs and writes COMPLETE only after verification.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import torch

torch.set_num_threads(4)
SRC = Path('/home/param/models/trained-weights/pose-duplex-first-band-20261030/checkpoints/pose-duplex-first-band-20261030/dcp-227')
OUT = Path('/home/param/outputs/pose-mono-dcp227-export')
host = socket.gethostname().split('.')[0]
ts = json.loads((SRC / 'train_state.json').read_text())
assert ts['step'] == int((SRC / 'COMPLETE').read_text()) == 227
assert ts['world'] == 40 and len(ts['rank_hosts']) == 40
assert ts['run_tag'] == 'pose-duplex-first-band-20261030'
ranks = [r for r, h in enumerate(ts['rank_hosts']) if h == host]
assert len(ranks) == 8, (host, ranks)
OUT.mkdir(parents=True, exist_ok=True)
manifest = {'source': str(SRC), 'host': host, 'step': 227, 'ranks': ranks,
            'cfg_flat_sha': ts['cfg_flat_sha'], 'weights_only': True, 'files': {}}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    return h.hexdigest()

gain = None
for name in [f'rank{r}.pt' for r in ranks] + (['rep.pt'] if 0 in ranks else []):
    start = time.time()
    state = torch.load(SRC / name, map_location='cpu', mmap=True, weights_only=True)
    model = state['model']
    assert model and all(isinstance(t, torch.Tensor) for t in model.values())
    tensors = {}
    for key, value in model.items():
        assert value.is_floating_point()
        dtype = torch.float32 if key == 'vgain.theta' else torch.bfloat16
        tensors[key] = value.to(dtype=dtype).contiguous().clone()
        assert torch.isfinite(tensors[key]).all(), key
        if key == 'vgain.theta' and value.numel():
            gain = float(value.item())
    tmp = OUT / (name + '.tmp')
    torch.save(tensors, tmp)
    tmp.replace(OUT / name)
    del tensors
    saved = torch.load(OUT / name, map_location='cpu', mmap=True, weights_only=True)
    assert saved.keys() == model.keys()
    for key, value in model.items():
        dtype = torch.float32 if key == 'vgain.theta' else torch.bfloat16
        assert saved[key].dtype == dtype
        assert torch.equal(saved[key], value.to(dtype=dtype)), (name, key)
    manifest['files'][name] = {'bytes': (OUT/name).stat().st_size,
                             'sha256': sha(OUT/name), 'tensors': len(saved),
                             'elements': sum(t.numel() for t in saved.values()),
                             'verified_against_source': True}
    print(json.dumps({'host': host, 'file': name, **manifest['files'][name],
                      'seconds': round(time.time()-start, 2)}), flush=True)
    del state, model, saved
    gc.collect()

if 0 in ranks:
    assert gain is not None
    cfg = dict(ts['cfg_view'])
    cfg.update(layers=64, tower_layers=36, dec_layers=28, proj_r=2048, tower_vocab=151682)
    meta = {'format': 'ppm-local-bf16-shards-v1', 'src': str(SRC), 'step': 227,
            'world': 40, 'hosts': ts['rank_hosts'], 'run_tag': ts['run_tag'],
            'cfg': cfg, 'pose': 1, 'decp_bb': 1, 'decp_bb_alpha': cfg['decp_bb_alpha'],
            'vgain_theta_fp32': gain, 'save_time': ts['save_time'],
            'source_cfg_flat_sha': ts['cfg_flat_sha'], 'weights_only': True,
            'export_note': 'Model tensors only; BF16 except exact FP32 vgain.theta. No optimizer.'}
    (OUT/'meta.json').write_text(json.dumps(meta, indent=2)+'\n')
    for name in ['meta.json']:
        manifest['files'][name] = {'bytes': (OUT/name).stat().st_size, 'sha256': sha(OUT/name)}

manifest['complete'] = True
(OUT/f'export-manifest-{host}.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({'host': host, 'complete': True, 'bytes': sum(v['bytes'] for v in manifest['files'].values())}), flush=True)
