"""
The wavefront sampler (WavefrontNet, generate_wavefront) against the frame-major one (StabiliserCachedNet,
cached_universal.generate) on csv rows' first FRAMES latent frames, each row at its seed, eager and compiled + graphed:
per-frame rel-L2 of the committed (t_h) and the resolved latents. Floors, changes that keep the math: the frame-major
sampler with the clip beside a copy of itself (batch 2, same seed) vs alone, which changes only GEMM shapes, and the
frame-major sampler compiled vs eager. YAML: a sampler.yaml with one checkpoint_path in both blocks.
One rollout per process, so they run in parallel:
  run YAML ROW FRAMES SAMPLER(frame|wave) BATCH COMPILE(0|1) OUT_DIR -> OUT_DIR/row<ROW>_<SAMPLER>_b<BATCH>_c<COMPILE>.safetensors
  compare OUT_DIR ROW...
"""

import csv
import gc
import os
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'production')]

from bimodal.cached_resolver import ResolverNet
from bimodal.sampler import RESOLVE_SEED_STRIDE, StabiliserCachedNet, WavefrontNet, build_modes, generate_wavefront
from cached_universal import audio_start, generate
from sink_universal import load_chunk
from utils.common import AUTOCAST_DTYPE

SAMPLERS = {'frame': (StabiliserCachedNet, generate), 'wave': (WavefrontNet, generate_wavefront)}
PAIRS = {
    'wave-frame eager': ('wave_b1_c0', 'frame_b1_c0'), 'wave-frame compiled': ('wave_b1_c1', 'frame_b1_c1'),
    'floor batch2-1 eager': ('frame_b2_c0', 'frame_b1_c0'), 'floor compiled-eager': ('frame_b1_c1', 'frame_b1_c0'),
}


def rel(a, b):
    """Per-frame ||a - b|| / ||b|| over frames 1.. of [C, L, h, w]."""
    return ((a - b).square().sum(dim=(0, 2, 3)).sqrt() / b.square().sum(dim=(0, 2, 3)).sqrt())[1:]


def rollout(net, sample, B, stabilising, resolver, clip, seed, config, compile):
    """
    One rollout of clip, alone (B = 1) or beside copies of itself, the sampler's resolve as on_commit.

    -> committed, resolved latents [C, L, h, w] of the first copy
    """
    stab, res, device = config['stabilising'], config['resolving'], 'cuda'
    latents, pose = clip[0], clip[1]
    cached = net(stabilising, latents.shape[2:], stab['substeps'], device, compile, B, stab['history_t'])
    a = (1 - res['t_start']) / (1 - stab['history_t'])
    sigma = (res['t_start'] ** 2 - (a * stab['history_t']) ** 2) ** 0.5
    ts = torch.linspace(res['t_start'], 0, res['num_steps'] + 1, device=device)
    poses = pose[None].expand(B, -1, -1, -1, -1).to(device)
    code_rows = clip[3][None].expand(B, -1, -1).to(device)
    resolved = torch.zeros_like(poses)
    resolved[:, :, 0] = latents[:, 0].to(device)
    C, _, h, w = latents.shape

    def resolve(f, x):
        eps = torch.stack([torch.randn(C, h, w, device=device, generator=torch.Generator(device=device).manual_seed(
            seed * RESOLVE_SEED_STRIDE + f)) for _ in range(B)])
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
            resolver.frame(resolved, poses, code_rows, f, a * x + sigma * eps, ts, cached.audio_cache[(audio_start(f),) * B])

    committed = sample(cached, [clip] * B, [seed] * B, device, compile, on_commit=resolve)[0]
    out = committed.clone(), resolved[0].clone()
    del cached
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run(path, row, frames, sampler, B, compile, out_dir):
    config = yaml.safe_load(open(path))
    assert config['stabilising']['checkpoint_path'] == config['resolving']['checkpoint_path'], 'one bimodal checkpoint'
    stabilising, resolving = build_modes(config, Path(__file__).parent, 'cuda')
    paths = list(csv.DictReader(open(config['csv'])))[row]['paths']
    clip = load_chunk(os.path.join(config['root_dir'], paths), config['pose_file'], frames, config['mimi_file'])
    seed = config['seed'] + row
    name = f'row{row}_{sampler}_b{B}_c{int(compile)}'
    print(f'{name}: {paths}, {clip[0].shape[1]} frames, seed {seed}', flush=True)
    committed, resolved = rollout(*SAMPLERS[sampler], B, stabilising, ResolverNet(resolving), clip, seed, config, compile)
    os.makedirs(out_dir, exist_ok=True)
    save_file({'committed': committed.contiguous().cpu(), 'resolved': resolved.contiguous().cpu()}, os.path.join(out_dir, f'{name}.safetensors'))


def compare(out_dir, rows):
    for row in rows:
        runs = {key: load_file(os.path.join(out_dir, f'row{row}_{key}.safetensors'))
                for key in {key for pair in PAIRS.values() for key in pair}}
        for kind in ('committed', 'resolved'):
            columns = {name: rel(runs[a][kind], runs[b][kind]) for name, (a, b) in PAIRS.items()}
            print(f'row {row} {kind:9s} | ' + ' | '.join(columns), flush=True)
            for f in range(len(next(iter(columns.values())))):
                print(f'{f + 1:15d} | ' + ' | '.join(f'{100 * c[f]:>{len(name) - 1}.3f}%' for name, c in columns.items()))
            print(f'{"mean":>15s} | ' + ' | '.join(f'{100 * c.mean():>{len(name) - 1}.3f}%' for name, c in columns.items()),
                  flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        path, row, frames, sampler, B, compile, out_dir = sys.argv[2:]
        run(path, int(row), int(frames), sampler, int(B), bool(int(compile)), out_dir)
    else:
        compare(sys.argv[2], [int(row) for row in sys.argv[3:]])
