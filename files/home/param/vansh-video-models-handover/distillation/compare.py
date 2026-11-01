import argparse
import bisect
import gc
import hashlib
import json
from pathlib import Path
import random
import tomllib

import torch


SCHEDULE_KEYS = ('band_steps', 'resolve_steps', 'history_t', 't_start')
PROBE_FRACTIONS = (0.1, 0.5, 0.9)
MODEL_PAIRS = (('speedy', 'dense'), ('dense', 'teacher'), ('speedy', 'teacher'))


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def select_windows(source, count, seed, root):
    counts = [max(0, (int(length) - 4) // 4) for length in source.lengths]
    prefix = [0]
    for number in counts:
        prefix.append(prefix[-1] + number)
    if not 0 < count <= prefix[-1]:
        raise ValueError(f'Requested {count} windows from {prefix[-1]} available full windows')
    windows = []
    for index, window in enumerate(random.Random(seed).sample(range(prefix[-1]), count)):
        clip = bisect.bisect_right(prefix, window) - 1
        record = source.clips[clip]
        windows.append({'clip_index': clip, 'path': str(Path(record.video_path).parent.relative_to(root)),
                        'uuid': record.uuid, 'start': 4 + 4 * (window - prefix[clip]),
                        'noise_seed': seed + index * 100003})
    return windows


def tensor_fingerprint(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def metrics(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    difference = a - b
    a_norm, b_norm, delta_norm = a.norm().item(), b.norm().item(), difference.norm().item()
    return {'rms': a.square().mean().sqrt().item(), 'reference_rms': b.square().mean().sqrt().item(),
            'difference_rms': difference.square().mean().sqrt().item(), 'max_abs': difference.abs().max().item(),
            'l2_norm': a_norm, 'reference_l2_norm': b_norm, 'difference_l2': delta_norm,
            'magnitude_ratio': a_norm / b_norm if b_norm else None,
            'relative_l2': delta_norm / b_norm if b_norm else None,
            'cosine': (a @ b).item() / (a_norm * b_norm) if a_norm and b_norm else None}


def comparison(outputs):
    pairs = {'speedy_vs_dense': (outputs['speedy'], outputs['dense']),
             'dense_vs_teacher': (outputs['dense'], outputs['teacher']),
             'speedy_vs_teacher': (outputs['speedy'], outputs['teacher']),
             'student_changes_vs_teacher': (outputs['speedy'] - outputs['teacher'],
                                            outputs['dense'] - outputs['teacher'])}
    return {name: {'all': metrics(a, b), 'frames': [metrics(a[:, :, j], b[:, :, j]) for j in range(4)],
                   'windows': [metrics(a[i], b[i]) for i in range(len(a))]} for name, (a, b) in pairs.items()}


class VelocityRecorder:
    def __init__(self, net, substeps, window_index, start):
        self.net, self.substeps, self.window_index, self.start = net, substeps, window_index, start
        self.calls, self.tags, self.values = 0, [], []

    def __enter__(self):
        self.original = self.net.forward_pass
        self.net.forward_pass = self.record
        return self

    def __exit__(self, *exc):
        self.net.forward_pass = self.original

    def record(self, frames, plan, ext, audio, audio_mask, heads, keep=()):
        call = self.calls
        self.calls += 1
        result = self.original(frames, plan, ext, audio, audio_mask, heads, keep)
        velocity = result[0]
        if velocity is None:
            return result
        batch = velocity.shape[0]
        round_index, substep = divmod(call - 2, self.substeps)
        round_index += 1
        for head, frame in enumerate(frames[:heads]):
            valid = torch.as_tensor(frame[4]).bool().expand(batch).cpu()
            times = torch.as_tensor(frame[2]).float().expand(batch).cpu()
            for row in valid.nonzero().flatten().tolist():
                slot = frame[3]
                if call < 2 or not 4 <= slot < 8:
                    raise ValueError('Recorder expects the two prefills followed by stabilizer-only rollout calls')
                absolute_frame = self.start - 4 + round_index + slot - 4
                self.tags.append({'window': self.window_index, 'batch_row': row, 'call': call, 'round': round_index,
                                  'substep': substep, 'head': head, 'slot': slot, 'time': times[row].item(),
                                  'frame': absolute_frame, 'generated': absolute_frame >= self.start, 'valid': True})
                self.values.append(velocity[row, :, head].detach().cpu())
        return result


def velocity_comparison(values, tags):
    if any(len(value) != len(tags) for value in values.values()):
        raise ValueError('Velocity records and tags differ in length')

    def pairs(indices):
        return {f'{a}_vs_{b}': metrics(values[a][indices], values[b][indices]) for a, b in MODEL_PAIRS}

    result = {'all': pairs(slice(None)), 'predictions': [dict(tag, comparison=pairs(i)) for i, tag in enumerate(tags)]}
    for field, columns in [('calls', ('window', 'call')), ('frames', ('window', 'frame')),
                            ('times', ('time',)), ('slots', ('slot',))]:
        groups = {}
        for index, tag in enumerate(tags):
            key = tuple(round(tag[column], 8) if column == 'time' else tag[column] for column in columns)
            groups.setdefault(key, []).append(index)
        result[field] = [dict(zip(columns, key), predictions=len(indices), comparison=pairs(indices))
                         for key, indices in groups.items()]
    generated = [index for index, tag in enumerate(tags) if tag.get('generated')]
    if generated:
        result['generated_frames_only'] = pairs(generated)
    return result


def common_probes(raw, config, seed):
    gt, h = raw['latents'], config['history_t']
    lower = (h + (1 - h) * torch.arange(4) / 4).clamp_min(config['stab_t_min'])
    upper = (h + (1 - h) * torch.arange(1, 5) / 4).clamp_max(config['stab_t_max'])
    if not (upper > lower).all():
        raise ValueError('Probe bands must have positive width')
    rng = torch.Generator().manual_seed(seed + 2147483647)
    probes = []
    for fraction in PROBE_FRACTIONS:
        t = (lower + fraction * (upper - lower))[None].expand(gt.shape[0], -1)
        noise = torch.randn(gt[:, :, 4:].shape, generator=rng)
        history_noise = torch.randn(gt[:, :, 1:4].shape, generator=rng)
        x = (1 - t[:, None, :, None, None]) * gt[:, :, 4:] + t[:, None, :, None, None] * noise
        history = (1 - h) * gt[:, :, 1:4] + h * history_noise
        probes.append((x, t, history))
    return probes


def snapshot(directory):
    from distillation.archive_checkpoint import inspect_checkpoint
    from distillation.eval_bundle import MARKER, read_bundle

    directory = Path(directory)
    if (directory / MARKER).exists():
        return read_bundle(directory)
    marker = json.loads((directory / 'checkpoint.complete.json').read_text())
    inspect_checkpoint(directory.absolute(), marker['checkpoint_id'])
    meta = json.loads((directory / 'metadata.json').read_text())
    if marker['checkpoint_id'] != meta['checkpoint_id']:
        raise ValueError(f'Checkpoint is still being written: {directory}')
    path = directory / 'student.safetensors'
    sizes = {item['path']: item['size'] for item in marker['files']}
    if path.name not in sizes or path.stat().st_size != sizes[path.name]:
        raise ValueError(f'Student checkpoint is incomplete: {directory}')
    return path, meta


def load_snapshot(net, state, lora):
    from distillation.lora import inject_lora

    if lora:
        inject_lora(net, **lora)
    parameters = dict(net.original_named_parameters())
    if parameters.keys() != state.keys():
        raise ValueError(f'Checkpoint keys differ: missing={parameters.keys() - state.keys()}, '
                         f'extra={state.keys() - parameters.keys()}')
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape:
                raise ValueError(f'Checkpoint shape differs: {name}')
            parameter.copy_(state[name])


def main():
    parser = argparse.ArgumentParser(description='Paired four-frame stabilizer rollouts with a reusable evaluation manifest.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--csv', required=True, help='explicit evaluation CSV; held-out status must be checked separately')
    parser.add_argument('--root-dir', help='override the data root for evaluation only')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--selection', help='optional explicit seed/windows JSON when creating the manifest')
    parser.add_argument('--windows', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dense')
    parser.add_argument('--speedy')
    parser.add_argument('--output')
    parser.add_argument('--prepare-only', action='store_true', help='create/validate window manifest on CPU, then exit')
    parser.add_argument('--compile', action='store_true', help='compile whole blocks; first run pays compilation')
    parser.add_argument('--check-merged', action='store_true', help='also measure LoRA merge rounding on identical rollout/probe inputs')
    args = parser.parse_args()
    if not args.prepare_only and not all((args.dense, args.speedy, args.output)):
        parser.error('--dense, --speedy and --output are required unless --prepare-only is used')

    from dataloading.ds_aloe import AloeChunkSource, ShuffledDenseWindows

    config = tomllib.loads(Path(args.config).read_text())
    root = Path(args.root_dir or config['root_dir']).resolve()
    source = AloeChunkSource(str(root), str(Path(args.csv).resolve().parent), Path(args.csv).name,
                             pose_file=config['pose_latents'], mimi_file=config['mimi_codes'], bad_audio=config['bad_audio'])
    signature = {'csv_sha256': sha256(args.csv), 'pose_latents': config['pose_latents'], 'mimi_codes': config['mimi_codes'],
                 'bad_audio': config['bad_audio'], 'schedule': {key: config[key] for key in SCHEDULE_KEYS},
                 'audio': config['model']['audio'], 'codebooks': config['model']['audio']['num_codebooks'],
                 'probe_fractions': list(PROBE_FRACTIONS), 'probe_t_bounds': [config['stab_t_min'], config['stab_t_max']]}
    manifest_path = Path(args.manifest)
    existing = manifest_path.exists()
    selection = json.loads(Path(args.selection).read_text()) if args.selection else None
    if selection is not None and (selection['seed'] != args.seed or len(selection['windows']) != args.windows):
        raise ValueError('Explicit selection must match --seed and --windows')
    if existing:
        manifest = json.loads(manifest_path.read_text())
        if manifest['signature'] != signature:
            raise ValueError('Evaluation CSV, conditioning, or rollout differs from the saved manifest')
        if manifest['seed'] != args.seed or len(manifest['windows']) != args.windows:
            raise ValueError('--seed or --windows differs from the saved manifest')
        if selection is not None:
            selected = [{key: window[key] for key in selection['windows'][0]} for window in manifest['windows']]
            if selected != selection['windows']:
                raise ValueError('Explicit selection differs from the saved manifest')
    else:
        manifest = {'version': 2, 'signature': signature, 'seed': args.seed,
                    'windows': selection['windows'] if selection is not None else select_windows(source, args.windows, args.seed, root)}
        if selection is not None:
            manifest['selection'] = {'path': str(Path(args.selection).resolve()), 'sha256': sha256(args.selection)}
    raw_rows = []
    for window in manifest['windows']:
        clip, start = window['clip_index'], window['start']
        if str(Path(source.clips[clip].video_path).parent.relative_to(root)) != window['path']:
            raise ValueError('Evaluation clip order differs from the saved manifest')
        ofi = torch.tensor([0, *range(start - 3, start + 4)])
        video0, video, pose0, pose, body_start, codes, shifted, _ = source.load_window(clip, ofi)
        raw = {'latents': ShuffledDenseWindows._gather(video0, video, body_start, ofi)[None].float(),
               'pose': ShuffledDenseWindows._gather(pose0, pose, body_start, ofi)[None].float(),
               'ofi': shifted[None], 'codes': codes[None],
               'drop': torch.tensor([bool(source.bad_audio[clip]) if source.bad_audio is not None else False])}
        digest = tensor_fingerprint(raw)
        if existing and window['conditioning_sha256'] != digest:
            raise ValueError(f'Evaluation tensors changed for {window["path"]}, frame {start}')
        window['conditioning_sha256'] = digest
        raw_rows.append(raw)
    if not existing:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if args.prepare_only:
        print(json.dumps({'manifest': str(manifest_path), 'windows': len(raw_rows)}), flush=True)
        return

    from distillation.bimodal_dmd import Rollout, UniformRollout, UniformStabilizerRollout, stab_pass
    from distillation.bimodal_net import audio_latents, build_net, load_mimi, load_state_dict
    from safetensors.torch import save_file

    output = Path(args.output)
    latent_path = output.with_suffix('.safetensors')
    if output.exists() or latent_path.exists():
        raise FileExistsError('Comparison outputs already exist; choose a new --output')
    dense_path, dense_meta = snapshot(args.dense)
    speedy_path, speedy_meta = snapshot(args.speedy)
    step = lambda meta: meta['counters']['warmup_step'] + meta['counters']['outer_step']
    if step(dense_meta) != step(speedy_meta):
        raise ValueError('Compare checkpoints at the same logical step, including critic warmup')
    for key in ('initial_checkpoint_sha256', 'audio', 'rollout', 'latent_hw'):
        if dense_meta['compatibility'][key] != speedy_meta['compatibility'][key]:
            raise ValueError(f'Paired checkpoint compatibility differs: {key}')
    teacher_path = Path(config['model']['checkpoint_path'])
    teacher_hash = sha256(teacher_path)
    if teacher_hash != dense_meta['compatibility']['initial_checkpoint_sha256']:
        raise ValueError('Evaluation teacher differs from the student initialization teacher')
    if config['model']['audio'] != dense_meta['compatibility']['audio']:
        raise ValueError('Evaluation audio architecture differs from training')
    for key in (*SCHEDULE_KEYS, 'pose_latents', 'mimi_codes', 'bad_audio'):
        if config[key] != dense_meta['compatibility']['rollout'][key]:
            raise ValueError(f'Evaluation rollout or conditioning differs from training: {key}')

    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    mimi, template = load_mimi(config['model']['audio'], device)
    with torch.no_grad():
        for raw in raw_rows:
            raw['audio'] = audio_latents(mimi, raw.pop('codes').to(device), signature['codebooks']).cpu()
    del mimi
    rollout = {'uniform': UniformRollout, 'uniform_stab': UniformStabilizerRollout, 'fused': Rollout}[config['rollout']](
        config['band_steps'], config['resolve_steps'], config['history_t'], config['t_start'])
    outputs, own_velocities, probe_velocities, own_tags, probe_tags = {}, {}, {}, None, None
    checkpoints = {'teacher': (teacher_path, None), 'dense': (dense_path, dense_meta), 'speedy': (speedy_path, speedy_meta)}
    if args.check_merged:
        checkpoints['speedy_merged'] = (speedy_path, speedy_meta)
    identities = {}
    for name, (path, meta) in checkpoints.items():
        state = load_state_dict(path)
        if name == 'speedy_merged':
            from distillation.export import merged_parameters
            state = merged_parameters(state, meta.get('lora'))
        base = {key: value for key, value in state.items() if not key.endswith(('.lora_A', '.lora_B'))}
        net = build_net(base, config['model']['audio'], tuple(raw_rows[0]['latents'].shape[-2:]), device, template)
        if meta is not None and name != 'speedy_merged':
            load_snapshot(net, state, meta.get('lora'))
        del state, base
        net.eval().requires_grad_(False)
        net.configure(args.compile, 'none', 'block')
        generated, own, probes, current_own_tags, current_probe_tags = [], [], [], [], []
        with torch.no_grad():
            for index, (raw, window) in enumerate(zip(raw_rows, manifest['windows'])):
                rows = {key: value.to(device) for key, value in raw.items()}
                rows['history_t'] = config['history_t']
                rng = torch.Generator().manual_seed(window['noise_seed'])
                noise = torch.randn(raw['latents'].shape, generator=rng).to(device)
                renoise = torch.randn(raw['latents'].shape, generator=rng).to(device)
                tokens = net.audio_tokens(rows['audio'])
                with VelocityRecorder(net, max(config['band_steps']), index, window['start']) as recorded:
                    y, _ = rollout(net, rows, tokens, noise, renoise, stab_only=True)
                if not torch.isfinite(y).all():
                    raise ValueError(f'Nonfinite stabilizer output: {name}, window {index}')
                generated.append(y.cpu())
                own.extend(recorded.values)
                current_own_tags.extend(recorded.tags)
                audio = net.frame_audio(tokens, rows['ofi'][:, 4:], rows['drop'])
                mask = torch.ones(len(tokens), 4, dtype=torch.bool, device=device)
                for probe, (x, t, history) in enumerate(common_probes(raw, config, window['noise_seed'])):
                    velocity = stab_pass(net, rows, x.to(device), t.to(device), history.to(device), audio, mask)
                    for head in range(4):
                        probes.append(velocity[0, :, head].cpu())
                        current_probe_tags.append({'window': index, 'call': probe, 'probe_fraction': PROBE_FRACTIONS[probe],
                                                   'head': head, 'slot': head + 4, 'time': t[0, head].item(),
                                                   'frame': window['start'] + head, 'generated': True, 'valid': True})
                print(json.dumps({'model': name, 'window': index + 1, 'windows': len(raw_rows)}), flush=True)
        outputs[name] = torch.cat(generated)
        own_velocities[name], probe_velocities[name] = torch.stack(own), torch.stack(probes)
        if not torch.isfinite(own_velocities[name]).all() or not torch.isfinite(probe_velocities[name]).all():
            raise ValueError(f'Nonfinite velocity prediction: {name}')
        if own_tags is not None and (own_tags != current_own_tags or probe_tags != current_probe_tags):
            raise ValueError('Velocity prediction times, slots or validity differ between models')
        own_tags, probe_tags = current_own_tags, current_probe_tags
        identities[name] = {'path': str(path), 'sha256': teacher_hash if name == 'teacher' else sha256(path)}
        if meta is not None:
            identities[name].update(checkpoint_id=meta['checkpoint_id'], counters=meta['counters'],
                                    lora=meta.get('lora'), merged=name == 'speedy_merged')
        # The recorder retains the network and its bound forward method. Drop
        # it as well and collect module/method cycles before loading the next
        # checkpoint, otherwise several complete networks can remain resident.
        del net, recorded
        gc.collect()
        torch.cuda.empty_cache()
    result = {'step': step(dense_meta), 'manifest': str(manifest_path.resolve()), 'manifest_sha256': sha256(manifest_path),
              'method': 'Same four-frame stabilizer rollout, GT seed history, fixed CPU Gaussian noise, pose, audio and codebooks; exact unmerged adapters; every real velocity head captured, pads excluded; controlled GT-noised velocity probes share x, t and history; no resolver.',
              'runtime': {'torch': torch.__version__, 'device': torch.cuda.get_device_name(device),
                          'compile': args.compile, 'allow_tf32': torch.backends.cuda.matmul.allow_tf32},
              'source_sha256': {name: sha256(Path(__file__).resolve().parents[1] / name) for name in
                                ('distillation/compare.py', 'distillation/bimodal_net.py',
                                 'distillation/bimodal_dmd.py', 'distillation/lora.py')},
              'checkpoints': identities, 'comparison': comparison(outputs),
              'trajectory_velocities': velocity_comparison(own_velocities, own_tags),
              'common_input_velocities': velocity_comparison(probe_velocities, probe_tags),
              'latent_outputs': str(latent_path.resolve())}
    if args.check_merged:
        result['merge_rounding'] = {
            'latents': metrics(outputs['speedy_merged'], outputs['speedy']),
            'trajectory_velocities': metrics(own_velocities['speedy_merged'], own_velocities['speedy']),
            'common_input_velocities': metrics(probe_velocities['speedy_merged'], probe_velocities['speedy']),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    tensors = {f'latents/{name}': value for name, value in outputs.items()}
    tensors.update({f'trajectory_velocities/{name}': value for name, value in own_velocities.items()})
    tensors.update({f'common_input_velocities/{name}': value for name, value in probe_velocities.items()})
    save_file(tensors, str(latent_path))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: value['all'] for name, value in result['comparison'].items()}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
