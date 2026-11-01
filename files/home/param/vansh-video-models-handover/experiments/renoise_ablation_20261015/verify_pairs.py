import json
import subprocess
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

root = Path('/mnt/paramfs/param/outputs/renoise-ablation-20261015')
manifest = json.loads((root / 'manifest.json').read_text())
results = []
for clip in manifest['clips']:
    n = clip['index']
    paths = [root / arm / f'sample_{n}_stabilised_latents.safetensors' for arm in ('baseline', 'direct')]
    ready = all((root / 'logs' / f'{arm}-{n}.log').read_text(errors='replace').count('Decoding latents with Wan2.2 VAE') >= 3 for arm in ('baseline', 'direct'))
    if not ready:
        continue
    for arm in ('baseline', 'direct'):
        exit_path = root / 'logs' / f'{arm}-{n}.exit'
        if exit_path.exists():
            assert exit_path.read_text().strip() == '0', (arm, n)
    baseline, direct = [load_file(p)['latents'] for p in paths]
    delta = baseline.astype(np.float64) - direct.astype(np.float64)
    result = dict(index=n, split=clip['split'], uuid=clip['uuid'], shape=list(baseline.shape),
                  stabilizer_bitwise_equal=bool(np.array_equal(baseline, direct)),
                  stabilizer_max_abs=float(np.abs(delta).max()),
                  stabilizer_relative_l2=float(np.linalg.norm(delta.ravel()) / np.linalg.norm(baseline.ravel())))
    del baseline, direct, delta
    for arm in ('baseline', 'direct'):
        video = root / arm / f'sample_{n}.mp4'
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames,r_frame_rate,duration', '-of', 'json', str(video)]))['streams'][0]
        result[arm + '_video'] = probe
    results.append(result)
    print(n, result['stabilizer_bitwise_equal'], result['stabilizer_max_abs'], flush=True)
report = dict(expected_pairs=16, completed_pairs=len(results), all_stabilizer_bitwise_equal=all(r['stabilizer_bitwise_equal'] for r in results), pairs=results)
(root / 'pair_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'pairs'}))
