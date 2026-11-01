"""
The resolving mode's cached velocity (ResolverNet) against the training path on the same windows, GT history, noise
and t, with the training path on each window alone vs inside the batch as the floor; then eager timings. TOML:
universal_wide (the easy resolver, ORDINAL counts its windows) or universal_wide_bimodal (ORDINAL counts the
stabilising windows its resolving windows come from). YAML: a sampler.yaml.
Usage: python bimodal/validate_resolver.py YAML TOML ROW ORDINAL...
"""

import os
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'production')]

from bimodal.cached_resolver import ResolverNet, context_frames
from bimodal.sampler import build_modes
from dataloading.ds_aloe import AloeChunkSource, ShuffledDenseWindows
from dataloading.ds_bimodal import BimodalWindows
from dataloading.sources import AUDIO_CODE_QUANTUM
from models.bimodal.wan_bimodal import BimodalWidePipeline, ResolvingView
from models.pose.wan_universal import UniversalWidePipeline
from utils.common import AUTOCAST_DTYPE


def load_clip(source, n):
    """
    -> latents, pose [C, L, h, w] fp32, codes [codebooks, T] long: clip n in full.
    """
    clip = source.clips[n]
    with safe_open(clip.mimi_path, framework='pt') as handle:
        codes = handle.get_tensor('codes').long()
    return source._slice_latents(clip.video_path, 0, clip.length).float(), source._slice_latents(clip.pose_path, 0, clip.length).float(), codes


def stats(a, b):
    diff = (a - b).abs()
    return f'max {diff.max():.3e} mean {diff.mean():.3e}'


@torch.no_grad()
def main():
    config = yaml.safe_load(open(sys.argv[1]))
    toml = tomllib.load(open(sys.argv[2], 'rb'))
    toml.update(toml.pop('stabilising', {}))
    row, ordinals = int(sys.argv[3]), [int(o) for o in sys.argv[4:]]
    models = build_modes(config, Path(__file__).parent, 'cuda')[1]
    net, H = ResolverNet(models), config['resolving']['num_history']
    bimodal = toml['model']['type'] == 'universal_wide_bimodal'

    csv_path = config['csv']
    source = AloeChunkSource(config['root_dir'], os.path.dirname(csv_path), os.path.basename(csv_path), pose_file=config['pose_file'],
                             mimi_file=config['mimi_file'])
    if bimodal:
        pipe = ResolvingView(BimodalWidePipeline(toml), **toml['resolving'])
        data = ShuffledDenseWindows(source, toml['num_history'], toml['num_present'], toml['history_dropout'], 0, 1, len(ordinals), 1)
    else:
        pipe = UniversalWidePipeline(toml)
        data = ShuffledDenseWindows(source, H, 1, toml['history_dropout'], 0, 1, len(ordinals), 1)
    ids = [int(data._prefix[row]) + o for o in ordinals]
    data._perm, data._perm_epoch = np.array(ids), 0
    if bimodal:  # the batch's ofi is shifted by the stabilising window's audio offset
        batch = BimodalWindows(data, H)[0][1]
        shifts = [max(0, int(data._window(0, i)[1].max()) - (AUDIO_CODE_QUANTUM // 2 - 1)) for i in ids]
        windows = [torch.where(o >= 1, o + a, o) for o, a in zip(batch['original_frame_indices'], shifts)]
    else:
        batch = data[0]
        windows = [data._window(0, i)[1] for i in ids]
    fs = [int(w[-1]) for w in windows]
    latents, pose, codes = load_clip(source, row)
    B, (C, _, h, w) = len(ids), latents.shape
    for b, (window, f) in enumerate(zip(windows, fs)):
        a = max(0, f - (AUDIO_CODE_QUANTUM // 2 - 1))
        index = [max(j, 0) for j in context_frames(f, H)] + [f]
        same = (window.tolist() == context_frames(f, H) + [f], torch.equal(batch['latents_low'][b, :, window >= 0].float(), latents[:, index][:, window >= 0]),
                torch.equal(batch['pose_latents_low'][b, :, window >= 0].float(), pose[:, index][:, window >= 0]),
                torch.equal(batch['codes'][b], codes[:, 2 * a:2 * a + AUDIO_CODE_QUANTUM]))
        print(f'ordinal {ordinals[b]}: training window {window.tolist()} shifted {batch["original_frame_indices"][b].tolist()} '
              f'a {a}; window / latents / pose / codes equal to the sampler\'s: {same}')

    noise = torch.randn(B, C, 1 + H + 1, h, w, generator=torch.Generator().manual_seed(0))
    for t in (0.4, 0.1):
        batch['t'] = torch.zeros(B, 1 + H + 1)
        batch['t'][:, -1] = t
        batch['noise'] = noise
        v_train = models.net(pipe.prepare_inputs(batch)[0])[:, :, -1].float()
        for b, f in enumerate(fs):
            alone = {k: v[b:b + 1] for k, v in batch.items()}
            alone['window_source'] = torch.zeros(1, dtype=torch.long)
            v_alone = models.net(pipe.prepare_inputs(alone)[0])[0, :, -1].float()
            ofi = context_frames(f, H)
            index = [max(j, 0) for j in ofi]
            with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
                context = net.context(latents[None, :, index].cuda(), pose[None, :, index].cuda(), torch.tensor([ofi], device='cuda'))
                audio = net.audio(codes[None].cuda(), f)
                x_t = ((1 - batch['t'][b, -1]) * latents[:, f] + batch['t'][b, -1] * noise[b, :, -1])[None].cuda()
                v_cached = net.velocity(x_t, pose[None, :, f].cuda(), f, batch['t'][b, -1].cuda() * 1000, context, audio)[0]
            print(f't {t} ordinal {ordinals[b]} (f {f}): mean|v| {v_train[b].abs().mean():.4f}; cached vs train {stats(v_cached, v_train[b])}; '
                  f'cached vs train alone {stats(v_cached, v_alone)}; floor (train alone vs batch) {stats(v_alone, v_train[b])}')

    f = fs[-1]
    ofi = context_frames(f, H)
    index = [max(j, 0) for j in ofi]
    for n in (1, 2, 4):
        lat, pos = latents[None, :, index].expand(n, -1, -1, -1, -1).cuda(), pose[None, :, index].expand(n, -1, -1, -1, -1).cuda()
        cod, x_t, p = codes[None].expand(n, -1, -1).cuda(), latents[None, :, f].expand(n, -1, -1, -1).cuda(), pose[None, :, f].expand(n, -1, -1, -1).cuda()
        o, t = torch.tensor([ofi] * n, device='cuda'), torch.tensor(400.0, device='cuda')
        timed = {}
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
            for name, fn in (('context', lambda: net.context(lat, pos, o)), ('audio', lambda: net.audio(cod, f))):
                fn()
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(10):
                    out = fn()
                torch.cuda.synchronize()
                timed[name] = 100 * (time.time() - start)
                timed[name + '_out'] = out
            context, audio = timed.pop('context_out'), timed.pop('audio_out')
            net.velocity(x_t, p, f, t, context, audio)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                net.velocity(x_t, p, f, t, context, audio)
            torch.cuda.synchronize()
            timed['velocity'] = 100 * (time.time() - start)
        per_frame = {N: timed['context'] + timed['audio'] + N * timed['velocity'] for N in (4, 8, 12)}
        print(f'batch {n}: ' + ', '.join(f'{k} {v:.1f} ms' for k, v in timed.items()) + '; per frame ' +
              ', '.join(f'N{N} {v:.0f} ms' for N, v in per_frame.items()) + f'; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')


if __name__ == '__main__':
    main()
