"""Memory-bounded preview using the ordinary wavefront and frozen teacher solver.

Run explicitly with ``python -m bimodal.streaming_sampler CONFIG.yaml``. GPU
stabilization, GPU resolution, and CPU media decoding are separate phases. This
is an optional preview backend, not a training or throughput benchmark path.
"""

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch
from safetensors.torch import load_file, save_file
import yaml

from bimodal.audio_tables import resolver_tokens
from bimodal.sampler import RESOLVE_SEED_STRIDE, generate_wavefront, side_by_side
from bimodal.streaming import StreamedResolverNet, StreamedWavefrontNet, build_models_cpu, release_edges
from inference.common import CKPT_PATH, FPS, decode_wan22_vae, resolve_path
from models.pose.settings import wan_5b_wide
from sink_universal import load_chunk
from utils.common import AUTOCAST_DTYPE


BASE_DIR = Path(__file__).resolve().parent
GIB = 2 ** 30


def configure_memory_budget(limit_gib, runtime_reserve_gib=1.0):
    """Bound PyTorch allocations, leaving explicit space for CUDA/cuBLAS state.

    CUDA has no portable hard per-process VRAM limit. The allocator limit plus
    sampled nvidia-smi process residency is reported so this is never presented
    as a measured six-GiB peak before GPU validation has taken place.
    """
    if not 0 < runtime_reserve_gib < limit_gib <= 6:
        raise ValueError('Require 0 < runtime reserve < preview budget <= 6 GiB')
    device = torch.device('cuda', 0)
    total = torch.cuda.get_device_properties(device).total_memory
    allocated_limit = int((limit_gib - runtime_reserve_gib) * GIB)
    torch.cuda.set_per_process_memory_fraction(allocated_limit / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    return {'requested_process_budget_gib': limit_gib, 'runtime_reserve_gib': runtime_reserve_gib,
            'allocator_limit_bytes': allocated_limit,
            'limit_scope': 'PyTorch allocator; runtime reservation is estimated, process residency is sampled'}


def process_gpu_mib():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, check=False)
    values = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) == 2 and parts[0] == str(os.getpid()):
            try:
                values.append(float(parts[1]))
            except ValueError:
                pass
    return max(values, default=None)


def memory_record(phase, budget_gib):
    torch.cuda.synchronize()
    resident = process_gpu_mib()
    record = {'phase': phase, 'time': time.time(), 'peak_allocated_gib': torch.cuda.max_memory_allocated() / GIB,
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / GIB, 'sampled_process_mib': resident}
    print(json.dumps(record), flush=True)
    if resident is not None and resident > budget_gib * 1024:
        raise RuntimeError(f'Preview exceeded its {budget_gib} GiB process budget: {resident / 1024:.3f} GiB')
    return record


def release_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def load_clip(config, rows, index):
    return load_chunk(str(Path(config['root_dir']) / rows[index]['paths']), config['pose_file'],
                      config['max_latent_frames'], config['mimi_file'])


@torch.no_grad()
def stabilize(config, rows, output, budget_gib):
    mode = config['stabilising']
    models = build_models_cpu({**config, **mode}, BASE_DIR)
    cached = None
    records = []
    try:
        for index in config['rows']:
            clip = load_clip(config, rows, index)
            if cached is None:
                cached = StreamedWavefrontNet(models, clip[0].shape[2:], mode['substeps'], 'cuda', mode['history_t'])
            elif tuple(clip[0].shape[2:]) != tuple(cached.buf_latent.shape[2:]):
                raise ValueError('All preview clips must share spatial geometry')
            generated = generate_wavefront(cached, [clip], [config['seed'] + index], 'cuda', False)[0]
            save_file({'latents': generated.cpu()}, str(output / f'sample_{index}_stabilised_latents.safetensors'))
            del generated
            record = memory_record(f'stabilize:{index}', budget_gib)
            record.update(block_passes=cached.blocks.passes, transferred_bytes=cached.blocks.transferred_bytes)
            records.append(record)
    finally:
        if cached is not None:
            cached.blocks.close()
            del cached
        release_edges(models)
        del models
        release_gpu()
    return records


@torch.no_grad()
def resolve(config, rows, output, budget_gib):
    mode, stab = config['resolving'], config['stabilising']
    a = (1 - mode['t_start']) / (1 - stab['history_t'])
    variance = mode['t_start'] ** 2 - (a * stab['history_t']) ** 2
    if variance < 0:
        raise ValueError('Resolver renoising variance is negative')
    sigma = variance ** .5
    models = build_models_cpu({**config, **mode}, BASE_DIR)
    resolver = StreamedResolverNet(models, 'cuda')
    ts = torch.linspace(mode['t_start'], 0, mode['num_steps'] + 1, device='cuda')
    records = []
    try:
        for index in config['rows']:
            clip = load_clip(config, rows, index)
            original, pose, table, codes = clip
            stabilized = load_file(str(output / f'sample_{index}_stabilised_latents.safetensors'))['latents'].cuda()
            if stabilized.shape != original.shape:
                raise ValueError(f'Stabilized latent geometry differs for row {index}')
            C, L, h, w = stabilized.shape
            if L > 375:
                raise ValueError('The low-memory preview currently supports prefixes of at most 375 latent frames')
            poses, code_rows = pose[None].cuda(), codes[None].cuda()
            resolved = torch.zeros_like(poses)
            resolved[:, :, 0] = original[:, 0].cuda()
            # load_chunk retains the full conditioning table separately from
            # its waveform prefix. The teacher's encoder is different from the
            # trained student's, so encode that same full table with the teacher.
            tokens = resolver_tokens(resolver, [table.cuda()], code_rows, 1, {})
            seed = config['seed'] + index
            for frame in range(1, L):
                eps = torch.randn(C, h, w, device='cuda', generator=torch.Generator(device='cuda').manual_seed(
                    seed * RESOLVE_SEED_STRIDE + frame))
                with torch.autocast('cuda', dtype=AUTOCAST_DTYPE, cache_enabled=False):
                    resolver.frame(resolved, poses, code_rows, frame, a * stabilized[:, frame][None] + sigma * eps,
                                   ts, tokens)
                if frame % 16 == 0:
                    print(f'[streaming] resolve row {index}: {frame}/{L - 1}', flush=True)
            save_file({'latents': resolved[0].cpu()}, str(output / f'sample_{index}_latents.safetensors'))
            del stabilized, poses, code_rows, resolved, tokens, eps
            record = memory_record(f'resolve:{index}', budget_gib)
            record.update(block_passes=resolver.blocks.passes, transferred_bytes=resolver.blocks.transferred_bytes)
            records.append(record)
    finally:
        resolver.blocks.close()
        del resolver, ts
        release_edges(models)
        del models
        release_gpu()
    return records


@torch.no_grad()
def decode(config, rows, output):
    # CPU decoding has its own phase and does not compete for GPU model memory.
    # Keep its FP32 weights and activations; CPU/GPU decoder rounding is a
    # separately reported backend difference, not a change to the BF16 sampler.
    from transformers import MimiModel
    from models.autoencoders.wanvae import Wan2_2_VAE

    mimi = MimiModel.from_pretrained(config['audio']['model_name']).eval().requires_grad_(False)
    vae = Wan2_2_VAE(vae_pth=resolve_path(f'{CKPT_PATH}/{wan_5b_wide.vae_checkpoint}', BASE_DIR),
                    dtype=torch.float32, device='cpu')
    for index in config['rows']:
        clip = load_clip(config, rows, index)
        latents, _, _, codes = clip
        audio = mimi.decode(codes[None, :, :2 * latents.shape[1]]).audio_values[0, 0].float().numpy()
        items = [(f'samples_{index}', latents)]
        for suffix in ('', '_stabilised'):
            items.append((f'sample_{index}{suffix}', load_file(str(output / f'sample_{index}{suffix}_latents.safetensors'))['latents']))
        for name, values in items:
            decode_wan22_vae(vae, values[None], str(output / f'{name}.mp4'), fps=FPS, audio_np=audio, tag='streaming')
        side_by_side(output / f'samples_{index}.mp4', output / f'sample_{index}.mp4', output / f'sample_{index}_sxs.mp4')


def validate_config(config):
    if config.get('type') != 'universal_wide_bimodal' or not config.get('wavefront'):
        raise ValueError('Streaming preview requires the bimodal wavefront sampler')
    if config.get('batch_size') != 1:
        raise ValueError('Streaming preview uses one clip at a time')
    if config.get('compile') or config.get('cuda_graph'):
        raise ValueError('Streaming preview requires compile: false and cuda_graph: false')
    if not 2 <= config['max_latent_frames'] <= 375:
        raise ValueError('Streaming preview requires a complete prefix of 2..375 latent frames')
    if config['stabilising']['num_history'] != 3 or config['stabilising']['num_present'] != 4:
        raise ValueError('Preview stabilizer requires three history frames and four present frames')
    if config['resolving']['num_present'] != 1 or config['resolving']['num_steps'] < 1:
        raise ValueError('Preview resolver requires one present frame and positive Euler step count')
    if any(key in config for key in ('refs', 'hidden')):
        raise ValueError('References and hidden conditioning are not supported by this preview backend')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--phase', choices=('all', 'stabilize', 'resolve', 'decode'), default='all')
    parser.add_argument('--gpu-budget-gib', type=float, default=6.)
    parser.add_argument('--cpu-threads', type=int, default=8)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    validate_config(config)
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    config_copy = output / Path(args.config).name
    if config_copy.resolve() != Path(args.config).resolve():
        shutil.copy2(args.config, config_copy)
    with Path(config['csv']).open() as handle:
        rows = list(csv.DictReader(handle))
    report = {'backend': 'streamed-one-block', 'compile': False, 'cuda_graph': False, 'pid': os.getpid(),
              'phase': args.phase, 'decoder_device': 'cpu', 'decoder_dtype': 'float32',
              'same_precision_as': 'ordinary eager wavefront and separate-checkpoint resolver',
              'gpu_validated': False, 'memory': []}
    start = time.monotonic()
    if args.phase != 'decode':
        report['budget'] = configure_memory_budget(args.gpu_budget_gib)
    try:
        if args.phase in ('all', 'stabilize'):
            report['memory'].extend(stabilize(config, rows, output, args.gpu_budget_gib))
        if args.phase in ('all', 'resolve'):
            report['memory'].extend(resolve(config, rows, output, args.gpu_budget_gib))
        if args.phase in ('all', 'decode'):
            decode(config, rows, output)
        report['complete'] = True
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        (output / f'streaming-{args.phase}.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
