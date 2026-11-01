"""Verify gathered weight-only export, then mark it loadable. Usage: script DIR."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

root = Path(sys.argv[1]).resolve()
hosts = ('landsend-2', 'landsend-3', 'landsend-4', 'landsend-5', 'landsend-12')
files = {}
ranks = []
for host in hosts:
    m = json.loads((root / f'export-manifest-{host}.json').read_text())
    assert m['complete'] and m['weights_only'] and m['step'] == 227
    assert m['cfg_flat_sha'] == '8f8d3445abcf8a7e'
    ranks += m['ranks']
    assert not set(files).intersection(m['files'])
    files.update(m['files'])
assert sorted(ranks) == list(range(40))
assert set(files) == {f'rank{r}.pt' for r in range(40)} | {'rep.pt', 'meta.json'}
verified = []
for name, info in sorted(files.items()):
    p = root / name
    assert p.stat().st_size == info['bytes'], name
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    assert h.hexdigest() == info['sha256'], name
    print(json.dumps({'verified': name, 'bytes': info['bytes'], 'sha256': info['sha256']}), flush=True)
    verified.append(name)
meta = json.loads((root/'meta.json').read_text())
assert meta['format'] == 'ppm-local-bf16-shards-v1'
assert meta['step'] == 227 and meta['world'] == 40
assert meta['vgain_theta_fp32'] == -5.5737104415893555
result = {'complete': True, 'step': 227, 'world': 40, 'weights_only': True,
          'bytes': sum(x['bytes'] for x in files.values()), 'sha256_verified': files,
          'verified_at': time.time()}
(root/'VERIFIED_EXPORT.json').write_text(json.dumps(result, indent=2)+'\n')
(root/'COMPLETE.tmp').write_text('227\n')
os.replace(root/'COMPLETE.tmp', root/'COMPLETE')
print(json.dumps({'COMPLETE': str(root), 'bytes': result['bytes'], 'files': len(files)}), flush=True)
