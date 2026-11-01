import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex

from inference.distilled import FIXED, MODES, validate_config


DEFAULT_SEED = 2026
DEFAULT_LATENT_FRAMES = 1875
WEIGHT_FILES = {'raw': 'student.safetensors', 'ema': 'model_ema.safetensors'}


def main():
    parser = argparse.ArgumentParser(description='Write a ready-to-run config for the step-2911 nine-reference student.')
    parser.add_argument('--root', type=Path, required=True, help='Handover asset/work directory from amuhandover.md')
    parser.add_argument('--sample-dir', type=Path, required=True, help='Prepared video, pose-v3 and local-Mimi clip directory')
    parser.add_argument('--reference-bank', type=Path, required=True, help='Directory containing actor UUID directories')
    parser.add_argument('--actor-id', required=True, help='Actor UUID for this clip, from the source CSV')
    parser.add_argument('--clip-key', required=True, help='Original CSV paths value; preserves deterministic reference selection')
    parser.add_argument('--name', required=True, help='New run name, e.g. clip-00 or clip-00-preflight')
    parser.add_argument('--weights', choices=WEIGHT_FILES, default='raw', help='Raw matches the comparison videos (default)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--latent-frames', type=int, default=DEFAULT_LATENT_FRAMES,
                        help='Require exactly this usable prefix; 1875 = 299.88 seconds, 33 = 5.16 seconds')
    args = parser.parse_args()
    if not args.name or args.name in ('.', '..') or Path(args.name).name != args.name:
        parser.error('--name must be a single directory name')
    root = args.root.expanduser().resolve()
    checkpoint = root / 'checkpoints' / 'full-gt' / 'step-002911' / WEIGHT_FILES[args.weights]
    config = {
        **FIXED, **deepcopy(MODES),
        'sample_dir': str(args.sample_dir.expanduser().resolve()),
        'output_dir': str(root / 'runs' / f'{args.name}-{args.weights}'),
        'seed': args.seed, 'max_latent_frames': args.latent_frames, 'minimum_latent_frames': args.latent_frames,
        'vae_path': str(root / 'assets' / 'wan22-5b-weights' / 'Wan2.2_VAE.pth'),
        'refs': {'bank': str(args.reference_bank.expanduser().resolve()),
                 'actor_id': args.actor_id, 'clip_key': args.clip_key},
    }
    config['audio']['model_name'] = str(root / 'assets' / 'mimi')
    for mode in ('stabilising', 'resolving'):
        config[mode]['checkpoint_path'] = str(checkpoint)
    validate_config(config)
    path = root / 'configs' / f'{args.name}-{args.weights}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(config, handle, indent=2)
        handle.write('\n')
    quoted = shlex.quote(str(path))
    print(f'Config: {path}')
    print(f'Check: python -m inference.distilled {quoted} --check')
    print(f'Run:   CUDA_VISIBLE_DEVICES=0 python -u -m inference.distilled {quoted}')


if __name__ == '__main__':
    main()
