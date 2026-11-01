import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zlib

import yaml


REQUIRED = {
    'sample_dir', 'output_dir', 'seed', 'batch_size', 'max_latent_frames', 'minimum_latent_frames',
    'fps', 'pose_file', 'mimi_file', 'type', 'compile', 'cuda_graph', 'wavefront', 'renoising',
    'stabilising', 'resolving', 'audio',
    'refs', 'vae_path', 'runtime_audio_guidance_scale',
}
FIXED = {
    'batch_size': 1, 'fps': 25, 'pose_file': 'pose_latents_electron_v3.safetensors',
    'mimi_file': 'mimi_local.safetensors', 'type': 'universal_wide_bimodal',
    'compile': True, 'cuda_graph': True, 'wavefront': True, 'renoising': False,
    'runtime_audio_guidance_scale': 1.0,
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


def safetensors_header(path):
    with Path(path).open('rb') as handle:
        prefix = handle.read(8)
        length = int.from_bytes(prefix, 'little')
        if len(prefix) != 8 or not 0 < length <= min(Path(path).stat().st_size - 8, 16 * 1024 * 1024):
            raise ValueError(f'Invalid safetensors header: {path}')
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise ValueError(f'Invalid safetensors header: {path}')
    payload = Path(path).stat().st_size - 8 - length
    for key, tensor in header.items():
        if key == '__metadata__':
            continue
        offsets = tensor.get('data_offsets', [])
        shape = tensor.get('shape', [])
        if (len(offsets) != 2 or not all(type(n) is int for n in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= payload
                or not all(type(n) is int and n >= 0 for n in shape)):
            raise ValueError(f'Invalid tensor header {key}: {path}')
    return header


def tensor_shape(path, key, dtypes):
    header = safetensors_header(path)
    if key not in header:
        raise ValueError(f'Missing tensor {key}: {path}')
    if header[key]['dtype'] not in dtypes:
        raise ValueError(f'Unexpected dtype for {key}: {header[key]["dtype"]} in {path}')
    return header[key]['shape']


def reference_seed(clip_key):
    return [1234, zlib.crc32(clip_key.encode()) & 0x7FFFFFFF]


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
        if name == 'audio':
            for filename in ('config.json', 'model.safetensors'):
                if not (path / filename).is_file():
                    raise ValueError(f'Missing local Mimi asset: {path / filename}')
    if config['stabilising']['checkpoint_path'] != config['resolving']['checkpoint_path']:
        raise ValueError('The stabilizer and resolver must share the same checkpoint_path')
    checkpoint = config['stabilising']['checkpoint_path']
    header = {key.removeprefix('model.diffusion_model.'): value for key, value in safetensors_header(checkpoint).items()}
    expected_refs = {'ref_patch_embedding.weight': [3840, 48, 1, 2, 2], 'ref_patch_embedding.bias': [3840]}
    for key, shape in expected_refs.items():
        if header.get(key, {}).get('shape') != shape:
            raise ValueError(f'Expected nine-reference student tensor {key} with shape {shape}: {checkpoint}')
    vae = absolute_path(config['vae_path'], 'vae_path')
    if not vae.is_file() or vae.name != 'Wan2.2_VAE.pth':
        raise ValueError(f'vae_path must name the existing Wan2.2_VAE.pth file: {vae}')
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
    floats = {'F32', 'F16', 'BF16'}
    video = tensor_shape(sample / 'video_latents.safetensors', 'latents', floats)
    pose = tensor_shape(sample / config['pose_file'], 'latents', floats)
    codes = tensor_shape(sample / config['mimi_file'], 'codes', {'I8', 'I16', 'I32', 'I64', 'U8', 'U16', 'U32', 'U64'})
    if len(video) != 4 or video[1] != 48 or min(video) <= 0 or pose != video:
        raise ValueError('Video and pose must have matching [L, 48, h, w] latent geometry')
    if any(size % 2 for size in video[2:]):
        raise ValueError('Latent height and width must be divisible by the 2 x 2 patch size')
    if len(codes) != 2 or codes[0] < config['audio']['num_codebooks']:
        raise ValueError('Local Mimi codes must have at least 16 codebooks')
    available = min(video[0], codes[1] // 2)
    length = min(available, maximum) if maximum is not None else available
    if length < minimum:
        raise ValueError(f'Only {length} usable latent frames; require {minimum}')
    refs = config['refs']
    require_keys(refs, {'bank', 'actor_id', 'clip_key'}, 'refs')
    bank = absolute_path(refs['bank'], 'refs.bank')
    actor = refs['actor_id']
    if not isinstance(actor, str) or not actor.strip() or Path(actor).name != actor or actor in ('.', '..'):
        raise ValueError('refs.actor_id must be a nonempty actor folder name')
    if not isinstance(refs['clip_key'], str) or not refs['clip_key'].strip() or refs['clip_key'] != refs['clip_key'].strip():
        raise ValueError('refs.clip_key must be the original nonempty CSV paths value without surrounding whitespace')
    if not (bank / actor).is_dir():
        raise ValueError(f'Missing actor reference folder: {bank / actor}')
    keep = []
    for slot in range(1, 10):
        path = bank / actor / f'property_{slot}' / 'latents.safetensors'
        keep.append(path.is_file())
        if not keep[-1]:
            continue
        shape = tensor_shape(path, 'video', floats)
        if len(shape) != 4 or shape[0] < 1 or shape[1:] != video[1:]:
            raise ValueError(f'References must have [N, 48, h, w] geometry matching the clip: {path}')
    if not any(keep):
        raise ValueError(f'Actor has no reference latents: {bank / actor}')
    output = absolute_path(config['output_dir'], 'output_dir')
    if output.exists() or output.is_symlink():
        raise ValueError(f'output_dir must be new; refusing to overwrite {output}')
    return {
        'checkpoint': checkpoint, 'shared_weights': True, 'reference_keep': keep,
        'reference_seed': reference_seed(refs['clip_key']), 'vae_path': str(vae),
        'stabilizer_band_steps': [2, 2, 2, 2], 'resolver_steps': 1,
        'runtime_audio_guidance_scale': config['runtime_audio_guidance_scale'], 'renoising': False,
        'latent_frames': length, 'decoded_frames': 4 * length - 3, 'output_dir': str(output),
    }


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def build_conditioned_modes(config, latents):
    import numpy as np
    import torch
    from bimodal import sampler
    from bimodal.graph_resolver import GraphResolver
    from dataloading.ds_aloe import load_actor_refs
    from utils.common import AUTOCAST_DTYPE

    stabilizing, resolving = sampler.build_modes(config, Path(__file__).resolve().parents[1] / 'bimodal', 'cuda')
    ref_config = config['refs']
    refs, keep = load_actor_refs(
        ref_config['bank'], ref_config['actor_id'], latents[:, :1],
        np.random.default_rng(reference_seed(ref_config['clip_key'])),
    )
    cached = sampler.WavefrontNet(
        stabilizing, latents.shape[2:], config['stabilising']['substeps'], 'cuda',
        config['compile'], config['batch_size'], config['stabilising']['history_t'],
        guidance_scale=config['runtime_audio_guidance_scale'],
    )
    cached.refs = (refs.transpose(0, 1)[None].cuda(), keep[None].cuda())
    resolver = GraphResolver(resolving, num_steps=config['resolving']['num_steps'])
    with torch.autocast('cuda', dtype=AUTOCAST_DTYPE, cache_enabled=False):
        resolver.set_refs(refs[None].cuda(), keep[None].cuda())
    return stabilizing, cached, resolver


def run(config):
    contract = validate_config(config)
    for command in ('ffmpeg', 'ffprobe'):
        if shutil.which(command) is None:
            raise RuntimeError(f'{command} is required on PATH')

    import torch
    from bimodal import sampler
    from bimodal.audio_tables import resolver_tokens
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
        **contract,
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
            stabilizing, cached, resolver = build_conditioned_modes(config, latents)
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
    parser.add_argument('--check', action='store_true', help='Validate local inputs and print the sampling contract without importing CUDA or writing files')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.check:
        print(json.dumps(validate_config(config), indent=2))
    else:
        run(config)


if __name__ == '__main__':
    main()
