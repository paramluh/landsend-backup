import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import yaml


REQUIRED = {
    'sample_dir', 'output_dir', 'seed', 'batch_size', 'max_latent_frames', 'minimum_latent_frames',
    'fps', 'pose_file', 'mimi_file', 'type', 'compile', 'cuda_graph', 'wavefront', 'renoising',
    'stabilising', 'resolving', 'audio',
}
FIXED = {
    'batch_size': 1, 'fps': 25, 'pose_file': 'pose_latents_electron_v3.safetensors',
    'mimi_file': 'mimi_local.safetensors', 'type': 'universal_wide_bimodal',
    'compile': True, 'cuda_graph': True, 'wavefront': True, 'renoising': False,
}
MODES = {
    'stabilising': {'num_history': 3, 'num_present': 4, 'history_t': .2, 'substeps': 2},
    'resolving': {'num_history': 2, 'num_present': 1, 't_start': .2, 'num_steps': 1},
    'audio': {
        'in_dim': 512, 'num_codebooks': 16, 'num_layers': 4, 'cross_attention_codes': 4,
        'drop': 2, 'injection_offset': 4, 'injection_interval': 3, 'max_blocks': 8,
    },
}


def require_keys(config, keys, name):
    if not isinstance(config, dict) or set(config) != keys:
        actual = set(config) if isinstance(config, dict) else set()
        raise ValueError(f'{name}: missing keys {sorted(keys - actual)}, unexpected keys {sorted(actual - keys)}')


def require_values(config, expected, name):
    for key, value in expected.items():
        if config[key] != value or type(config[key]) is not type(value):
            raise ValueError(f'{name}.{key} must be {value!r}, got {config[key]!r}')


def absolute_path(value, name):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f'{name} must be an absolute path')
    return Path(value)


def validate_config(config):
    require_keys(config, REQUIRED, 'config')
    require_values(config, FIXED, 'config')
    for name, expected in MODES.items():
        path_key = 'model_name' if name == 'audio' else 'checkpoint_path'
        require_keys(config[name], set(expected) | {path_key}, name)
        require_values(config[name], expected, name)
        path = absolute_path(config[name][path_key], f'{name}.{path_key}')
        if not (path.is_dir() if name == 'audio' else path.is_file()):
            raise ValueError(f'{name}.{path_key} does not exist: {path}')
    if config['stabilising']['checkpoint_path'] == config['resolving']['checkpoint_path']:
        raise ValueError('The distilled stabilizer and resolver require separate student checkpoints')
    if type(config['seed']) is not int:
        raise ValueError('seed must be an integer')
    minimum, maximum = config['minimum_latent_frames'], config['max_latent_frames']
    if type(minimum) is not int or minimum < 2:
        raise ValueError('minimum_latent_frames must be an integer >= 2')
    if maximum is not None and (type(maximum) is not int or maximum < minimum):
        raise ValueError('max_latent_frames must be null or an integer >= minimum_latent_frames')
    sample = absolute_path(config['sample_dir'], 'sample_dir')
    for filename in ('video_latents.safetensors', config['pose_file'], config['mimi_file']):
        if not (sample / filename).is_file():
            raise ValueError(f'Missing input: {sample / filename}')
    output = absolute_path(config['output_dir'], 'output_dir')
    if output.exists() or output.is_symlink():
        raise ValueError(f'output_dir must be new; refusing to overwrite {output}')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run(config):
    validate_config(config)
    for command in ('ffmpeg', 'ffprobe'):
        if shutil.which(command) is None:
            raise RuntimeError(f'{command} is required on PATH')

    import torch
    from bimodal import sampler
    from bimodal.audio_tables import resolver_tokens
    from bimodal.graph_resolver import GraphResolver
    from distillation.latent_export import persist_latents
    from inference.bounded_decode import decode_wan22_vae_bounded
    from utils.common import AUTOCAST_DTYPE

    if not torch.cuda.is_available():
        raise RuntimeError('Distilled inference requires a CUDA GPU')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    started = time.monotonic()
    status = {
        'status': 'loading', 'pid': os.getpid(), 'seed': config['seed'],
        'stabilizer_steps_per_slot': config['stabilising']['substeps'],
        'stabilizer_total_steps': config['stabilising']['substeps'] * config['stabilising']['num_present'],
        'resolver_steps': config['resolving']['num_steps'], 'renoising': config['renoising'],
        'compile': config['compile'], 'cuda_graph': config['cuda_graph'], 'wavefront': config['wavefront'],
    }

    def report(**updates):
        status.update(updates, elapsed_seconds=round(time.monotonic() - started, 2))
        write_json(output / 'status.json', status)
        print(json.dumps(status), flush=True)

    report()
    try:
        with torch.no_grad():
            clip = sampler.load_chunk(
                config['sample_dir'], config['pose_file'], config['max_latent_frames'], config['mimi_file'],
            )
            latents, pose, code_table, codes = clip
            if latents.ndim != 4 or latents.shape[0] != 48 or pose.shape != latents.shape:
                raise ValueError('Video and pose must have matching [48, L, h, w] loaded latent geometry')
            if codes.ndim != 2 or codes.shape[0] < config['audio']['num_codebooks']:
                raise ValueError('Local Mimi codes must have at least 16 codebooks')
            length = latents.shape[1]
            if length < config['minimum_latent_frames']:
                raise ValueError(f'Only {length} usable latent frames; require {config["minimum_latent_frames"]}')
            report(latent_frames=length, decoded_frames=4 * length - 3,
                   duration_seconds=(4 * length - 3) / config['fps'])
            stabilizing, resolving = sampler.build_modes(config, Path(__file__).resolve().parents[1] / 'bimodal', 'cuda')
            cached = sampler.WavefrontNet(
                stabilizing, latents.shape[2:], config['stabilising']['substeps'], 'cuda',
                config['compile'], config['batch_size'], config['stabilising']['history_t'],
            )
            resolver = GraphResolver(resolving, num_steps=config['resolving']['num_steps'])
            schedule = torch.linspace(config['resolving']['t_start'], 0, config['resolving']['num_steps'] + 1, device='cuda')
            poses, code_rows = pose[None].cuda(), codes[None].cuda()
            resolver_tables, audio_cache = [code_table.cuda()], {}
            resolved = torch.zeros_like(poses)
            resolved[:, :, 0] = latents[:, 0].cuda()
            torch.cuda.synchronize()
            report(status='sampling', completed_latent_frames=0, loading_seconds=round(time.monotonic() - started, 2))
            sampling_started = time.monotonic()
            steady_started = None

            def resolve(frame, committed):
                nonlocal steady_started
                tokens = resolver_tokens(resolver, resolver_tables, code_rows, frame, audio_cache)
                with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
                    resolver.frame(resolved, poses, code_rows, frame, committed, schedule, tokens)
                if frame == 1 or frame % 25 == 0 or frame == length - 1:
                    torch.cuda.synchronize()
                    now = time.monotonic()
                    if steady_started is None:
                        steady_started = now
                        status['first_frame_seconds_including_compile_capture'] = round(now - sampling_started, 2)
                    steady_elapsed = now - steady_started
                    report(
                        completed_latent_frames=frame, sampling_seconds=round(now - sampling_started, 2),
                        steady_sampling_seconds=round(steady_elapsed, 2),
                        estimated_remaining_seconds=(round(steady_elapsed / (frame - 1) * (length - 1 - frame), 1)
                                                     if frame > 1 else None),
                    )

            stabilised = sampler.generate_wavefront(
                cached, [clip], [config['seed']], 'cuda', config['cuda_graph'], on_commit=resolve,
            )[0]
            torch.cuda.synchronize()
            report(status='saving_latents', sampling_seconds=round(time.monotonic() - sampling_started, 2),
                   peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            receipts = persist_latents(output, [('resolved', resolved[0]), ('stabilised', stabilised)])
            report(status='decoding', latent_files=receipts)
            decoding_started = time.monotonic()
            audio = stabilizing.mimi.decode(codes[None, :, :2 * length].cuda()).audio_values[0, 0].float().cpu().numpy()
            decode_wan22_vae_bounded(stabilizing.vae, resolved, output / 'video.mp4', config['fps'], audio, 'distilled')
            probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration,size', '-of', 'json', str(output / 'video.mp4')],
                check=True, capture_output=True, text=True,
            )
            video = json.loads(probe.stdout)['format']
            if abs(float(video['duration']) - status['duration_seconds']) > .25:
                raise ValueError(f'Unexpected encoded video duration: {video}')
            report(status='complete', video=str(output / 'video.mp4'), encoded_video=video,
                   decoding_seconds=round(time.monotonic() - decoding_started, 2),
                   peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    except Exception as error:
        report(status='failed', error=f'{type(error).__name__}: {error}')
        raise


def main():
    parser = argparse.ArgumentParser(description='Compiled, cached full-wavefront 8-step stabilizer + 1-step resolver inference')
    parser.add_argument('config', help='YAML config with absolute input, checkpoint, Mimi and fresh output paths')
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()))


if __name__ == '__main__':
    main()
