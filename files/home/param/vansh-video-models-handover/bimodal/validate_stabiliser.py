"""
The stabilising mode's cached steps (sampler.py's StabiliserCachedNet), replayed on each window alone, against the
training path (dense and flex attention) on the same windows, noise and t: history at history_t, present slot j at the
sampler's band-j t. Floor: the training path on each window alone vs inside the batch. Ordinal 0 is the clip's startup
window; a late one has an audio offset > 0. TOML: universal_wide_stabiliser or universal_wide_bimodal.
YAML: a sampler.yaml.
Usage: python bimodal/validate_stabiliser.py YAML TOML ROW ORDINAL...
"""

import os
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'production')]

from bimodal.sampler import StabiliserCachedNet
from cached_universal import audio_start
from dataloading.ds_aloe import AloeChunkSource, ShuffledDenseWindows
from inference.common import sliding_code_table
from inference.loader import build_models
from models.bimodal.wan_bimodal import BimodalWidePipeline
from models.bimodal.stabiliser import StabiliserWidePipeline
from sink_universal import load_chunk

PIPELINES = {'universal_wide_stabiliser': StabiliserWidePipeline, 'universal_wide_bimodal': BimodalWidePipeline}


def stats(a, b):
    diff = (a - b).abs()
    return f'max {diff.max():.3e} mean {diff.mean():.3e}'


@torch.no_grad()
def main():
    config = yaml.safe_load(open(sys.argv[1]))
    config = {**config, **config['stabilising']}
    toml = tomllib.load(open(sys.argv[2], 'rb'))
    toml.update(toml.pop('stabilising', {}))
    row, ordinals = int(sys.argv[3]), [int(o) for o in sys.argv[4:]]
    device = 'cuda'
    models = build_models(config, Path(__file__).parent, device)
    pipe = PIPELINES[toml['model']['type']](toml)
    transformer_layers = list(models.net.layers[1:-1])
    H, P, m = models.num_history, models.num_present, config['substeps']

    csv_path = config['csv']
    source = AloeChunkSource(config['root_dir'], os.path.dirname(csv_path), os.path.basename(csv_path), pose_file=config['pose_file'],
                             mimi_file=config['mimi_file'])
    data = ShuffledDenseWindows(source, H, P, toml['history_dropout'], 0, 1, len(ordinals), 1)
    ids = [int(data._prefix[row]) + o for o in ordinals]
    data._perm, data._perm_epoch = np.array(ids), 0
    batch = data[0]
    windows = [data._window(0, i)[1] for i in ids]
    latents, pose, code_table, codes = load_chunk(os.path.dirname(source.clips[row].video_path), config['pose_file'], None)
    L = latents.shape[1]
    cached = StabiliserCachedNet(models, latents.shape[2:], m, device, False, 1, config['history_t'])
    rows, commits = cached.schedule(L)
    B, (C, _, h, w) = len(ids), latents.shape

    for b, window in enumerate(windows):
        real = window >= 0
        f_last = int(window.max())
        same = (torch.equal(batch['latents_low'][b, :, real].float(), latents[:, window[real]]),
                torch.equal(batch['pose_latents_low'][b, :, real].float(), pose[:, window[real]]),
                torch.equal(batch['codes'][b], sliding_code_table(code_table, codes, f_last)[0][0]))
        print(f'ordinal {ordinals[b]}: training window {window.tolist()} shifted {batch["original_frame_indices"][b].tolist()} '
              f'a {audio_start(f_last)}; latents / pose / codes equal to the sampler\'s: {same}')

    ofi = batch['original_frame_indices']
    noise = torch.randn(B, C, 1 + H + P, h, w, generator=torch.Generator().manual_seed(0))
    for s in (0, m - 1):
        t = torch.zeros(B, 1 + H + P)
        t[:, 1:1 + H] = torch.where(ofi[:, 1:1 + H] >= 0, torch.tensor(config['history_t']), torch.tensor(0.0))
        t[:, 1 + H:] = cached.t_band[torch.arange(P) * m + s] * batch['present_valid']
        batch['t'], batch['noise'] = t, noise
        v_train, v_alone = {}, {}
        for flex in (False, True):
            for layer in transformer_layers:
                layer.flex = flex
            v_train[flex] = models.net(pipe.prepare_inputs(batch)[0]).float()
            for b in range(B):
                alone = {k: v[b:b + 1] for k, v in batch.items()}
                alone['window_source'] = torch.zeros(1, dtype=torch.long)
                v_alone[flex, b] = models.net(pipe.prepare_inputs(alone)[0])[0].float()
        t_gpu = t.to(device)[:, None, :, None, None]
        x_t = (1 - t_gpu) * batch['latents_low'].to(device).float() + t_gpu * noise.to(device)
        poses = batch['pose_latents_low'].to(device).float()
        for b, window in enumerate(windows):
            first = int(torch.nonzero(window[1 + H:] >= 0)[0])
            q0 = int(window[1 + H + first]) - first
            cached.start_clip(x_t[b:b + 1, :, 0], cuda_graph=False)
            for i in range(1, 1 + H):
                if window[i] >= 1:
                    cached.buf_latent.copy_(x_t[b:b + 1, :, i])
                    cached.buf_pose.copy_(poses[b:b + 1, :, i])
                    cached.buf_store.copy_(commits[int(window[i]) - 1])
                    cached.commit()
            report = []
            for j in range(first, P):
                k, slot = q0 + j, 1 + H + j
                f_last = cached.audio_f_last(k, j, L)
                cached.set_audio([code_table.to(device)], [codes.to(device)], [f_last],
                                 torch.tensor([min(k, L - 1) - audio_start(f_last)], device=device))
                cached.buf_latent.copy_(x_t[b:b + 1, :, slot])
                cached.buf_pose.copy_(poses[b:b + 1, :, slot])
                cached.buf_ints.copy_(rows[(k - 1) * P * m + (P - 1 - j) * m + s])
                v_cached, ks, vs = cached.moving()
                cached._store(cached.buf_ints[cached.num_ctx:cached.num_ctx + 1] + cached.offsets, ks, vs)
                for flex in (False, True):
                    train, alone = v_train[flex][b, :, slot], v_alone[flex, b][:, slot]
                    report.append(f'  slot {j} (f {k}, t {t[b, slot]:.3f}) {"flex " if flex else "dense"}: '
                                  f'mean|v| {train.abs().mean():.4f}; cached vs train {stats(v_cached[0], train)}; '
                                  f'cached vs train alone {stats(v_cached[0], alone)}; floor {stats(alone, train)}')
            print(f's {s} ordinal {ordinals[b]} (q0 {q0}):\n' + '\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
