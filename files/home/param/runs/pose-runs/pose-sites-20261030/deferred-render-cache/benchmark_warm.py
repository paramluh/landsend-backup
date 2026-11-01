"""Compare canonical cold output with repeated warm runs and switched references."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml
from safetensors.torch import load_file

from warm_video import WarmVideo, REPO


def compare(expected, actual):
    result = {}
    for name in ('resolved', 'stabilised'):
        a = load_file(str(expected / f'{name}_latents.safetensors'))
        b = load_file(str(actual / f'{name}_latents.safetensors'))
        assert a.keys() == b.keys()
        key = next(iter(a))
        x, y = a[key].float(), b[key].float()
        assert x.shape == y.shape and torch.isfinite(y).all()
        delta = x - y
        result[name] = {'exact': torch.equal(x, y), 'max_abs': delta.abs().max().item(),
                        'rms': delta.square().mean().sqrt().item(), 'shape': list(x.shape)}
        # This optimization is buffer lifetime only: require exact model output.
        assert result[name]['exact'], (name, result[name])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline_config', type=Path)
    p.add_argument('out', type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    a = yaml.safe_load(args.baseline_config.read_text())
    original = Path(a['output_dir'])
    b = copy.deepcopy(a)
    # Deliberately change only reference identity. This isolates reference-cache
    # invalidation; the fixture is never published as a user-facing video.
    assets = json.loads(Path('/home/param/datasets/pose-demo/prompts/heldout_01/video_assets.json').read_text())
    b['refs'] = {'bank': assets['reference_bank'], 'actor_id': assets['actor_id'], 'clip_key': assets['clip_key']}
    b['max_latent_frames'] = b['minimum_latent_frames'] = 28
    b['output_dir'] = str(args.out / 'b-cold')
    bp = args.out / 'b-cold.yaml'
    bp.write_text(yaml.safe_dump(b))
    subprocess.run([sys.executable, '-u', '-m', 'inference.distilled', str(bp)], cwd=REPO, check=True)
    engine = WarmVideo()
    results = {}
    for name, cfg, reference in [('a-cold', a, original), ('b-warm', b, args.out / 'b-cold'),
                                  ('a-warm-after-b', a, original)]:
        cfg = copy.deepcopy(cfg)
        cfg['output_dir'] = str(args.out / name)
        started = time.monotonic()
        engine.runner.run(cfg)
        results[name] = {'wall_seconds': time.monotonic() - started,
                         'mode_setup': engine.last_mode_setup,
                         'parity': compare(reference, Path(cfg['output_dir'])),
                         'status': json.loads((Path(cfg['output_dir']) / 'status.json').read_text())}
        (args.out / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
        print('BENCHMARK_PASS', name, results[name]['wall_seconds'], flush=True)
    print('ALL_EXACT', flush=True)


if __name__ == '__main__':
    main()
