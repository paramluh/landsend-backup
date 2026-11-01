import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib

import yaml

from distillation.compare import sha256, snapshot
from distillation.export import export_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def validate_inputs(config, root, rows, indices, frames):
    from safetensors import safe_open

    checked = []
    for index in indices:
        directory = Path(root) / rows[index]['paths']
        files = [('video_latents.safetensors', 'latents'), (config['pose_latents'], 'latents'), (config['mimi_codes'], 'codes')]
        shapes, hashes = [], {}
        for filename, key in files:
            path = directory / filename
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                shapes.append(handle.get_slice(key).get_shape())
            hashes[filename] = sha256(path)
        video, pose, codes = shapes
        if len(video) != 4 or video[1:] != pose[1:] or min(video[0], pose[0]) < frames:
            raise ValueError(f'Video/pose geometry or complete prefix length differs: row {index}')
        if len(codes) != 2 or codes[0] < config['model']['audio']['num_codebooks'] or codes[1] < max(750, 2 * frames):
            raise ValueError(f'Insufficient Mimi codebooks, audio-table context or waveform codes: row {index}')
        checked.append({'row': index, 'video_shape': video, 'pose_shape': pose, 'codes_shape': codes, 'sha256': hashes})
    return checked


def sampler_config(config, csv_path, root, manifest, checkpoint, output, frames, resolver_steps):
    if len(set(config['band_steps'])) != 1:
        raise ValueError('The production wavefront sampler requires equal band_steps')
    return {'csv': str(Path(csv_path).resolve()), 'root_dir': str(Path(root).resolve()),
            'pose_file': config['pose_latents'], 'mimi_file': config['mimi_codes'],
            'rows': list(dict.fromkeys(window['clip_index'] for window in manifest['windows'])),
            'output_dir': str(Path(output).resolve()), 'type': 'universal_wide_bimodal',
            'seed': manifest['seed'], 'batch_size': 1, 'max_latent_frames': frames,
            'compile': True, 'cuda_graph': True, 'wavefront': True,
            'stabilising': {'checkpoint_path': str(Path(checkpoint).resolve()), 'num_history': 3, 'num_present': 4,
                           'history_t': config['history_t'], 'substeps': config['band_steps'][0]},
            'resolving': {'checkpoint_path': str(Path(config['model']['checkpoint_path']).resolve()),
                          'num_history': 2, 'num_present': 1, 't_start': config['t_start'], 'num_steps': resolver_steps},
            'audio': config['model']['audio']}


def video_command(gt, dense, speedy, output):
    return ['ffmpeg', '-y', '-i', str(gt), '-i', str(dense), '-i', str(speedy),
            '-filter_complex', '[0:v][1:v][2:v]hstack=inputs=3:shortest=1[v]', '-map', '[v]', '-map', '2:a?',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
            '-c:a', 'copy', str(output)]


def render_files(rows):
    return [name for index in rows for name in
            (f'samples_{index}.mp4', f'sample_{index}.mp4', f'sample_{index}_stabilised.mp4',
             f'sample_{index}_sxs.mp4', f'sample_{index}_latents.safetensors',
             f'sample_{index}_stabilised_latents.safetensors')]


def render_identity(config, checkpoint_id, inputs):
    sources = ('bimodal/sampler.py', 'bimodal/cached_resolver.py', 'bimodal/audio_tables.py',
               'inference/loader.py', 'inference/lora.py', 'distillation/lora.py',
               'production/cached_universal.py', 'production/sink_universal.py',
               'models/pose/wan_universal.py', 'models/pose/audio.py', 'models/pose/rope.py', 'models/wan/model.py')
    return {'config': config, 'checkpoint_id': checkpoint_id, 'inputs': inputs,
            'source_sha256': {name: sha256(ROOT / name) for name in sources}}


def write_render_receipt(config, checkpoint_id, inputs):
    directory = Path(config['output_dir'])
    files = [{'path': name, 'size': (directory / name).stat().st_size, 'sha256': sha256(directory / name)}
             for name in render_files(config['rows'])]
    if any(item['size'] == 0 for item in files):
        raise ValueError('Rendered output contains an empty file')
    receipt = {**render_identity(config, checkpoint_id, inputs), 'files': files}
    (directory / 'render.complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def verify_render_receipt(config, checkpoint_id, inputs):
    directory = Path(config['output_dir'])
    receipt = json.loads((directory / 'render.complete.json').read_text())
    identity = render_identity(config, checkpoint_id, inputs)
    if {key: receipt[key] for key in identity} != identity:
        raise ValueError('Existing rendered checkpoint, inputs, settings or inference source differs')
    if [item['path'] for item in receipt['files']] != render_files(config['rows']):
        raise ValueError('Existing render receipt has incomplete or unexpected output entries')
    for item in receipt['files']:
        path = directory / item['path']
        if path.stat().st_size != item['size'] or sha256(path) != item['sha256']:
            raise ValueError(f'Existing rendered file differs from its completion receipt: {path}')


def main():
    parser = argparse.ArgumentParser(description='Prepare or render identical GT | dense | speedy videos with one frozen teacher resolver.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dense', required=True)
    parser.add_argument('--speedy', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-latent-frames', type=int, required=True)
    parser.add_argument('--resolver-steps', type=int, required=True)
    parser.add_argument('--rows', type=int, nargs='+', help='optional subset of manifest clip rows, in explicit order')
    parser.add_argument('--backend', choices=('standard', 'streamed'), default='standard')
    parser.add_argument('--lora-mode', choices=('unmerged', 'merged'), default='unmerged',
                        help='exact saved adapters (primary comparison) or a separate rounded deployment export')
    parser.add_argument('--devices', type=int, nargs='+', help='explicit GPU IDs for concurrent standard sampler shards')
    parser.add_argument('--reuse-complete', action='store_true', help='reuse an arm only after verifying its complete render receipt')
    parser.add_argument('--run', action='store_true', help='run the prepared GPU inference commands')
    args = parser.parse_args()
    if args.max_latent_frames < 8 or args.resolver_steps < 1:
        parser.error('Require at least 8 latent frames and a positive resolver step count')
    if args.devices is not None and (args.backend != 'standard' or len(set(args.devices)) != len(args.devices)
                                     or min(args.devices) < 0):
        parser.error('--devices requires distinct nonnegative GPU IDs and the standard backend')
    config = tomllib.loads(Path(args.config).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    available_rows = list(dict.fromkeys(window['clip_index'] for window in manifest['windows']))
    selected_rows = available_rows if args.rows is None else args.rows
    if len(set(selected_rows)) != len(selected_rows) or not set(selected_rows) <= set(available_rows):
        raise ValueError('--rows must be distinct clip rows from the paired-window manifest')
    render_manifest = {**manifest, 'windows': [window for index in selected_rows for window in manifest['windows']
                                              if window['clip_index'] == index]}
    if sha256(args.csv) != manifest['signature']['csv_sha256']:
        raise ValueError('Video evaluation CSV differs from the paired-window manifest')
    for key in ('pose_latents', 'mimi_codes'):
        if config[key] != manifest['signature'][key]:
            raise ValueError(f'Video conditioning differs from the paired-window manifest: {key}')
    if config['model']['audio'] != manifest['signature']['audio']:
        raise ValueError('Video audio architecture differs from the paired-window manifest')
    if {key: config[key] for key in manifest['signature']['schedule']} != manifest['signature']['schedule']:
        raise ValueError('Video stabilizer schedule differs from the paired-window manifest')
    snapshots = {arm: snapshot(directory) for arm, directory in [('dense', args.dense), ('speedy', args.speedy)]}
    metas = [meta for _, meta in snapshots.values()]
    steps = [meta['counters']['warmup_step'] + meta['counters']['outer_step'] for meta in metas]
    if len(set(steps)) != 1:
        raise ValueError('Video checkpoints must share the same logical step')
    for key in ('initial_checkpoint_sha256', 'audio', 'rollout', 'latent_hw'):
        if metas[0]['compatibility'][key] != metas[1]['compatibility'][key]:
            raise ValueError(f'Video checkpoint compatibility differs: {key}')
    with Path(args.csv).open() as handle:
        rows = list(csv.DictReader(handle))
    for window in manifest['windows']:
        if rows[window['clip_index']]['paths'].strip() != window['path']:
            raise ValueError('Video clip identity differs from the paired-window manifest')
        if window['start'] + 4 > args.max_latent_frames:
            raise ValueError('Rendered prefix would omit a numerical evaluation window')
    inputs = validate_inputs(config, args.root_dir, rows, selected_rows, args.max_latent_frames)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands, configs, identities = {}, {}, {}
    pending = []
    for arm, (path, meta) in snapshots.items():
        checkpoint = output / f'{arm}-merged.safetensors' if meta.get('lora') and args.lora_mode == 'merged' else path
        configs[arm] = sampler_config(config, args.csv, args.root_dir, render_manifest, checkpoint, output / arm,
                                     args.max_latent_frames, args.resolver_steps)
        if meta.get('lora') and args.lora_mode == 'unmerged':
            configs[arm]['stabilising']['lora'] = meta['lora']
        config_path = output / f'{arm}.yaml'
        if args.backend == 'streamed':
            configs[arm].update(compile=False, cuda_graph=False)
            commands[arm] = [sys.executable, '-m', 'bimodal.streaming_sampler', str(config_path)]
        else:
            commands[arm] = [sys.executable, str(ROOT / 'bimodal/sampler.py'), str(config_path)]
        identities[arm] = {'checkpoint_id': meta['checkpoint_id'], 'source': str(path.resolve()),
                           'lora': meta.get('lora'), 'inference_checkpoint': str(checkpoint.resolve()),
                           'lora_mode': args.lora_mode if meta.get('lora') else None}
    selected = configs['dense']['rows']
    plan = {'step': steps[0], 'backend': args.backend,
            'devices': args.devices,
            'manifest_sha256': sha256(args.manifest), 'csv_sha256': sha256(args.csv),
            'checkpoints': identities, 'commands': commands, 'panel_order': ['ground_truth', 'dense', 'speedy'],
            'clips': [{'row': index, 'path': rows[index]['paths'], 'seed': manifest['seed'] + index} for index in selected],
            'inputs': inputs,
            'resolver': configs['dense']['resolving'], 'max_latent_frames': args.max_latent_frames,
            'method': 'Full open-loop clips; common per-clip GPU noise seeds; identical frozen teacher resolver and renoising seeds; '
                      + ('LoRA factors remain unmerged in their saved dtypes, with training fused-QKV arithmetic.'
                         if args.lora_mode == 'unmerged' else 'LoRA exported with explicit fp32 merge and base-dtype rounding.')}
    plan_path = output / 'render-plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Existing render plan differs; choose a new output directory')
    for arm, contents in configs.items():
        (output / f'{arm}.yaml').write_text(yaml.safe_dump(contents, sort_keys=False))
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    for command in commands.values():
        print(shlex.join(command), flush=True)
    if not args.run:
        return
    if sha256(config['model']['checkpoint_path']) != metas[0]['compatibility']['initial_checkpoint_sha256']:
        raise ValueError('Frozen video resolver differs from the training teacher')
    for arm, (path, meta) in snapshots.items():
        if meta.get('lora') and args.lora_mode == 'merged':
            checkpoint = Path(identities[arm]['inference_checkpoint'])
            if checkpoint.exists():
                exported = json.loads(checkpoint.with_suffix('.json').read_text())
                if exported['checkpoint_id'] != meta['checkpoint_id'] or exported['output_sha256'] != sha256(checkpoint):
                    raise ValueError('Existing merged checkpoint does not match this render plan')
            else:
                export_checkpoint(path.parent, checkpoint)
        destination = output / arm
        if destination.exists() and any(destination.glob('sample_*')):
            if not args.reuse_complete:
                raise FileExistsError(f'Rendered output already exists: {destination}')
            verify_render_receipt(configs[arm], meta['checkpoint_id'], inputs)
        else:
            pending.append(arm)
    if args.devices is None:
        for arm in pending:
            subprocess.run(commands[arm], cwd=ROOT, check=True)
    else:
        jobs = [(arm, shard) for arm in pending for shard in range(len(selected))]

        def render_worker(device, queue):
            for arm, shard in queue:
                command = commands[arm] + ['--shard', str(shard), '--num-shards', str(len(selected))]
                with (output / f'{arm}-shard-{shard}.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(device)},
                                   stdout=log, stderr=subprocess.STDOUT, check=True)

        with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
            futures = [pool.submit(render_worker, device, jobs[index::len(args.devices)])
                       for index, device in enumerate(args.devices)]
            for future in futures:
                future.result()
    for arm in pending:
        write_render_receipt(configs[arm], snapshots[arm][1]['checkpoint_id'], inputs)
    paired = output / 'paired'
    paired.mkdir(exist_ok=True)
    for index in selected:
        for suffix in ('', '_stabilised'):
            command = video_command(output / 'dense' / f'samples_{index}.mp4',
                                    output / 'dense' / f'sample_{index}{suffix}.mp4',
                                    output / 'speedy' / f'sample_{index}{suffix}.mp4',
                                    paired / f'sample_{index}{suffix}.mp4')
            subprocess.run(command, check=True)
    print(json.dumps({'paired_videos': str(paired), 'clips': len(selected), 'step': steps[0]}), flush=True)


if __name__ == '__main__':
    main()
