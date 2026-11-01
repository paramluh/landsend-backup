"""Real-checkpoint loader and cached-velocity comparison against canonical BimodalNet.

No training is run. Uses the explicit paired evaluation manifest and original clip
indices, and keeps cached-layout rounding separate from adapter loading errors.
"""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import torch
from torch import nn

from bimodal.sampler import WavefrontNet
from dataloading.ds_aloe import AloeChunkSource, ShuffledDenseWindows
from distillation.bimodal_dmd import stab_pass
from distillation.bimodal_net import audio_latents, build_net, load_state_dict
from distillation.compare import common_probes, load_snapshot, metrics, sha256, snapshot, tensor_fingerprint
from distillation.render_pairs import sampler_config
from inference.loader import Layered, build_models
from models.pose.wan_universal import FinalLayer, InitialLayer, TransformerLayer


def canonical_models(net, mimi):
    tr = net.transformer
    freqs = tr.freqs.to(net.slot_freqs.device)
    initial = InitialLayer(tr, net.audio_encoder, net.pose_patch_embedding, net.pad_token,
                           3, 4, net.K, 0., False)
    layers = [initial] + [TransformerLayer(block, net.audio_blocks[i], freqs, 3, 4, False, False)
                          for i, block in enumerate(tr.blocks)] + [FinalLayer(tr)]
    return SimpleNamespace(net=Layered(layers).eval(), mimi=mimi, num_history=3, num_present=4, num_codebooks=16)


def loaded_parameters(models):
    layers = models.net.layers
    holder = nn.Module()
    holder.transformer = layers[-1].unpatchify.__self__
    holder.audio_encoder = layers[0].audio_encoder
    holder.audio_adapter = nn.Module()
    holder.audio_adapter.blocks = nn.ModuleList([layer.audio_block for layer in layers[1:-1]
                                                 if layer.audio_block is not None])
    holder.pose_patch_embedding = layers[0].pose_patch_embedding
    holder.pad_token = layers[0].pad_token
    return {name.removeprefix('transformer.'): value for name, value in holder.named_parameters()}


def cached_probe(cached, rows, x, t, history, audio):
    cached.start_clip(rows['latents'][:, :, 0], False)
    context = [1]  # fixed sink entry
    for frame in range(1, 4):
        _, keys, values = cached._frame(history[:, :, frame - 1], rows['pose'][:, :, frame], False,
                                       cached.history_t, cached.anchor_freqs, cached.sink_ctx,
                                       cached.history_freqs, None, head=False)
        entries = list(cached.history_entries(frame))
        cached._store(torch.tensor(entries, device=x.device)[None] + cached.offsets, keys, values)
        context.append(entries[frame - 1])
    cached.wave_latent.copy_(x)
    cached.wave_pose.copy_(rows['pose'][:, :, 4:])
    cached.wave_valid.fill_(True)
    cached.wave_audio.copy_(audio)
    cached.wave_ints.copy_(torch.tensor([*context, 0], device=x.device))
    cached.t_table.index_copy_(0, cached.slot_steps, (t * 1000).flatten())
    return cached._wave()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=[0, 8])
    parser.add_argument('--compile-wave', action='store_true', help='also compare compiled wavefront and captured replay')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = tomllib.loads(Path(args.config).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    if sha256(args.csv) != manifest['signature']['csv_sha256']:
        raise ValueError('Evaluation CSV differs')
    windows = [window for window in manifest['windows'] if window['clip_index'] in args.rows]
    if set(args.rows) != {window['clip_index'] for window in windows}:
        raise ValueError('Requested rows missing from manifest')
    source = AloeChunkSource(args.root_dir, str(Path(args.csv).resolve().parent), Path(args.csv).name,
                             pose_file=config['pose_latents'], mimi_file=config['mimi_codes'], bad_audio=config['bad_audio'])
    raw_rows = []
    for window in windows:
        clip, start = window['clip_index'], window['start']
        ofi = torch.tensor([0, *range(start - 3, start + 4)])
        v0, video, p0, pose, body, codes, shifted, _ = source.load_window(clip, ofi)
        raw = {'latents': ShuffledDenseWindows._gather(v0, video, body, ofi)[None].float(),
               'pose': ShuffledDenseWindows._gather(p0, pose, body, ofi)[None].float(),
               'ofi': shifted[None], 'codes': codes[None],
               'drop': torch.tensor([bool(source.bad_audio[clip]) if source.bad_audio is not None else False])}
        if tensor_fingerprint(raw) != window['conditioning_sha256']:
            raise ValueError('Evaluation conditioning differs')
        raw_rows.append(raw)
    path, meta = snapshot(args.checkpoint)
    inference_config = sampler_config(config, args.csv, args.root_dir, manifest, path, Path(args.output).parent, 64, 2)
    mode_config = {**inference_config, **inference_config['stabilising']}
    if meta.get('lora'):
        mode_config['lora'] = meta['lora']
    models = build_models(mode_config, Path(__file__).resolve().parents[1] / 'bimodal', 'cuda')
    state = load_state_dict(path)
    base = {key: value for key, value in state.items() if not key.endswith(('.lora_A', '.lora_B'))}
    net = build_net(base, config['model']['audio'], tuple(raw_rows[0]['latents'].shape[-2:]),
                    'cuda', models.mimi.decoder_transformer)
    load_snapshot(net, state, meta.get('lora'))
    net.eval().requires_grad_(False).configure(False, 'none')
    actual, expected = loaded_parameters(models), dict(net.original_named_parameters())
    if actual.keys() != expected.keys():
        raise ValueError('Inference and canonical parameter keys differ')
    mismatches = [name for name in expected if actual[name].dtype != expected[name].dtype
                  or not torch.equal(actual[name], expected[name])]
    if mismatches:
        raise ValueError(f'Inference parameter values/dtypes differ: {mismatches}')
    del actual, expected, state, base
    gc.collect()
    torch.cuda.empty_cache()
    canonical, cached, reconstructed, compiled_outputs, captured_outputs = [], [], [], [], []
    with torch.no_grad():
        for raw in raw_rows:
            raw['audio'] = audio_latents(models.mimi, raw.pop('codes').cuda(), config['model']['audio']['num_codebooks']).cpu()
        inputs = []
        for raw, window in zip(raw_rows, windows):
            rows = {key: value.cuda() for key, value in raw.items()}
            rows['history_t'] = config['history_t']
            tokens = net.audio_tokens(rows['audio'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loaded_tokens = models.net.layers[0].audio_encoder(rows['audio'])
            if not torch.equal(loaded_tokens, tokens):
                raise ValueError('Loaded and canonical audio encoder predictions differ')
            audio = net.frame_audio(tokens, rows['ofi'][:, 4:], rows['drop'])
            mask = torch.ones(1, 4, dtype=torch.bool, device='cuda')
            for x, t, history in common_probes(raw, config, window['noise_seed']):
                values = (rows, x.cuda(), t.cuda(), history.cuda(), audio)
                canonical.append(stab_pass(net, rows, values[1], values[2], values[3], audio, mask).cpu())
                inputs.append(values)
        shape = tuple(raw_rows[0]['latents'].shape[-2:])
        cached_loaded = WavefrontNet(models, shape, 2, 'cuda', False, 1, config['history_t'])
        cached_reference = WavefrontNet(canonical_models(net, models.mimi), shape, 2, 'cuda', False, 1, config['history_t'])
        compiled = torch.compile(cached_loaded._wave, dynamic=False) if args.compile_wave else None
        graph = None
        for index, values in enumerate(inputs):
            left = cached_probe(cached_loaded, *values)
            right = cached_probe(cached_reference, *values)
            if not torch.equal(left, right):
                raise ValueError(f'Canonical reconstructed and loaded cached predictions differ: {metrics(left, right)}')
            cached.append(left.cpu())
            reconstructed.append(right.cpu())
            if compiled is not None:
                compiled_value = compiled().clone()
                if graph is None:
                    captured_value = torch.empty_like(compiled_value)
                    graph = cached_loaded._capture(lambda: captured_value.copy_(compiled()))
                graph.replay()
                if not torch.equal(captured_value, compiled_value):
                    raise ValueError(f'Captured replay differs: {metrics(captured_value, compiled_value)}')
                compiled_outputs.append(compiled_value.cpu())
                captured_outputs.append(captured_value.cpu())
            print(json.dumps({'probe': index + 1, 'probes': len(inputs), 'loader_exact': True}), flush=True)
    stacked = torch.cat(cached)
    report = {'checkpoint_id': meta['checkpoint_id'], 'checkpoint_sha256': sha256(path), 'lora': meta.get('lora'),
              'manifest_sha256': sha256(args.manifest), 'rows': args.rows, 'probes': len(inputs),
              'parameter_values_and_dtypes_exact': True, 'audio_encoder_exact': True,
              'loaded_vs_canonical_cached': metrics(stacked, torch.cat(reconstructed)),
              'cached_vs_canonical_full_window': metrics(stacked, torch.cat(canonical)),
              'method': 'Independent real-checkpoint loaders; exact parameters/dtypes and cached predictions; '
                        'full-window canonical BimodalNet compared separately to cached layout on identical x/t/history/audio.',
              'compile_wave': args.compile_wave, 'torch': torch.__version__,
              'source_sha256': {name: sha256(Path(__file__).resolve().parents[1] / name) for name in
                                ('distillation/check_inference_lora.py', 'distillation/bimodal_net.py',
                                 'distillation/lora.py', 'inference/lora.py', 'inference/loader.py',
                                 'production/cached_universal.py', 'bimodal/sampler.py')}}
    if compiled_outputs:
        report['compiled_vs_eager_cached'] = metrics(torch.cat(compiled_outputs), stacked)
        report['captured_vs_compiled'] = metrics(torch.cat(captured_outputs), torch.cat(compiled_outputs))
        report['compiled_vs_canonical_full_window'] = metrics(torch.cat(compiled_outputs), torch.cat(canonical))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
