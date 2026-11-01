"""Render one full gallery clip with the joint student's trained 2222 + one-step schedule.

Both modes share the same dense weights and deterministic actor references.
Only the initial sink is ground truth; each mode uses its own generated history.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import time


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(checkpoint, manifest_path, clip_id, weights, vae_path):
    checkpoint, manifest_path, vae_path = map(Path, (checkpoint, manifest_path, vae_path))
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    complete = json.loads((checkpoint / 'checkpoint.complete.json').read_text())
    if complete['checkpoint_id'] != metadata['checkpoint_id']:
        raise ValueError('Checkpoint completion and metadata identities differ')
    name = {'raw': 'student.safetensors', 'ema': 'model_ema.safetensors'}[weights]
    for required in ('metadata.json', name):
        recorded = next(item for item in complete['files'] if item['path'] == required)
        if (checkpoint / required).stat().st_size != recorded['size']:
            raise ValueError(f'Checkpoint file size differs: {required}')
    identity = metadata['compatibility']
    training = identity['training']
    if identity['schema'] != 'joint-nine-ref-v2-paired-batches' or metadata.get('lora') is not None:
        raise ValueError('Expected the dense paired joint nine-reference checkpoint')
    if (training['band_steps'] != [2, 2, 2, 2] or training['history_t'] != .2
            or training['resolver_steps'] != 1 or training['model']['num_refs'] != 9):
        raise ValueError('Checkpoint was not trained with the 2222 + one-step nine-reference contract')
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'joint-nine-ref-gallery-inputs-v1':
        raise ValueError('Unsupported gallery input manifest')
    selected = next(item for item in manifest['clips'] if item['global_id'] == clip_id)
    if selected['seed'] != 2026 + clip_id:
        raise ValueError('Clip seed differs from the existing gallery')
    if digest(manifest['csv']) != manifest['csv_sha256']:
        raise ValueError('Gallery CSV changed after selection')
    with Path(manifest['csv']).open(newline='') as handle:
        row = list(csv.DictReader(handle))[selected['csv_row']]
    if (row['paths'], row['uuid']) != (selected['paths'], selected['uuid']):
        raise ValueError('Selected clip does not match its gallery CSV row')
    if (manifest['root_dir'] != training['root_dir'] or manifest['pose_file'] != training['pose_latents']
            or manifest['mimi_file'] != training['mimi_codes']
            or manifest['reference_bank'] != identity['data']['reference_bank']):
        raise ValueError('Inference inputs differ from the checkpoint conditioning contract')
    for item in selected['files'].values():
        if Path(item['path']).stat().st_size != item['bytes'] or not os.access(item['path'], os.R_OK):
            raise ValueError(f'Clip input changed or is unreadable: {item["path"]}')
    if vae_path.name != 'Wan2.2_VAE.pth' or not vae_path.is_file():
        raise ValueError('Provide the existing Wan2.2_VAE.pth file')
    config = {
        'type': 'universal_wide_bimodal', 'audio': training['model']['audio'],
        'refs': {'bank': manifest['reference_bank']},
        'stabilising': {'checkpoint_path': str(checkpoint / name), 'num_history': 3,
                       'num_present': 4, 'history_t': .2, 'substeps': 2},
        'resolving': {'checkpoint_path': str(checkpoint / name), 'num_history': 2,
                     'num_present': 1, 't_start': .2, 'num_steps': 1},
        'runtime_audio_guidance_scale': 1., 'renoising': False, 'vae_path': str(vae_path),
    }
    return metadata, manifest, selected, config


def decode_saved(args, output, provenance):
    import torch
    from safetensors.torch import load_file
    from transformers import MimiModel
    from inference.bounded_decode import decode_wan22_vae_bounded
    from models.autoencoders.wanvae import Wan2_2_VAE

    generation_path = output / f'generation-complete-{args.clip}.json'
    generated = json.loads(generation_path.read_text())
    for key in ('checkpoint_id', 'manifest_sha256', 'weights', 'preflight', 'config', 'clip', 'runner_sha256'):
        if generated[key] != provenance[key]:
            raise ValueError(f'Decode request differs from completed generation: {key}')
    name = f'sample_{args.clip}' + ('_stabilised' if args.decode_only == 'stabilised' else '')
    latent_path = output / f'{name}_latents.safetensors'
    latent = next(item for item in generated['latent_files'] if item['path'] == str(latent_path))
    if digest(latent_path) != latent['sha256']:
        raise ValueError('Saved generated latent content changed')
    receipt_path = output / f'decode-complete-{args.clip}-{args.decode_only}.json'
    if receipt_path.exists():
        raise ValueError(f'Completed decode already exists: {receipt_path}')
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    started = time.monotonic()
    values = load_file(str(latent_path))['latents']
    if list(values.shape) != latent['shape'] or not torch.isfinite(values).all():
        raise ValueError('Saved generated latent geometry or finiteness changed')
    codes_path = provenance['clip']['files']['mimi_local.safetensors']['path']
    if digest(codes_path) != generated['input_sha256']['mimi_local.safetensors']:
        raise ValueError('Mimi source codes changed since generation')
    codes = load_file(codes_path)['codes'][:, :2 * values.shape[1]].long()
    mimi = MimiModel.from_pretrained(provenance['config']['audio']['model_name']).eval().requires_grad_(False).cuda()
    audio = mimi.decode(codes[None].cuda()).audio_values[0, 0].float().cpu().numpy()
    del mimi, codes
    torch.cuda.empty_cache()
    vae = Wan2_2_VAE(vae_pth=str(args.vae_path), dtype=torch.float32, device='cuda')
    video = output / f'{name}.mp4'
    decode_wan22_vae_bounded(vae, values[None], video, 25, audio, 'joint')
    result = dict(provenance, status='completed', mode=args.decode_only,
                  generation_receipt_sha256=digest(generation_path), latent=latent,
                  decoded_frames=4 * values.shape[1] - 3, video_duration_seconds=(4 * values.shape[1] - 3) / 25,
                  video={'path': str(video), 'bytes': video.stat().st_size, 'sha256': digest(video)},
                  elapsed_seconds=time.monotonic() - started)
    write_json(receipt_path, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--clip', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vae-path', type=Path, required=True)
    parser.add_argument('--weights', choices=('raw', 'ema'), default='raw')
    parser.add_argument('--decode-stabilised', action='store_true', help='Also decode the saved noisy .2 endpoint')
    parser.add_argument('--decode-ground-truth', action='store_true', help='Also decode GT and render a side-by-side')
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('--generate-only', action='store_true', help='Save both generated latent trajectories and exit')
    stage.add_argument('--decode-only', choices=('resolved', 'stabilised'), help='Decode a verified saved trajectory')
    stage.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--preflight', action='store_true', help='Five latent frames, including media serialization')
    args = parser.parse_args()
    metadata, manifest, selected, config = prepare(
        args.checkpoint, args.manifest, args.clip, args.weights, args.vae_path)
    output = args.output / ('preflight' if args.preflight else 'full') / args.weights
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / f'worker-complete-{args.clip}.json'
    if receipt_path.exists():
        raise ValueError(f'Completed output already exists: {receipt_path}')
    provenance = {
        'schema': 'joint-nine-ref-inference-v1', 'checkpoint': str(args.checkpoint),
        'checkpoint_id': metadata['checkpoint_id'], 'counters': metadata['counters'], 'weights': args.weights,
        'checkpoint_metadata_sha256': digest(args.checkpoint / 'metadata.json'),
        'manifest': str(args.manifest), 'manifest_sha256': digest(args.manifest),
        'clip': selected, 'config': config, 'preflight': args.preflight,
        'decode_stabilised': args.decode_stabilised, 'decode_ground_truth': args.decode_ground_truth,
        'host': socket.gethostname(), 'pid': os.getpid(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'runner_sha256': digest(__file__), 'status': 'prepared',
    }
    if args.decode_only:
        return decode_saved(args, output, provenance)
    write_json(output / f'config-{args.clip}.json', provenance)
    print(json.dumps({'event': 'prepared', 'clip': args.clip, 'checkpoint_id': metadata['checkpoint_id'],
                      'config': config, 'output': str(output)}), flush=True)
    if args.prepare_only:
        return

    import torch
    from bimodal import sampler
    from bimodal.audio_tables import resolver_tokens
    from bimodal.graph_resolver import GraphResolver
    from dataloading.ds_aloe import AloeChunkSource
    from distillation.latent_export import persist_latents
    from inference.bounded_decode import decode_wan22_vae_bounded
    from inference import loader
    from utils.common import AUTOCAST_DTYPE

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    started = time.monotonic()
    frame_limit = 5 if args.preflight else None
    clip = sampler.load_chunk(str(Path(manifest['root_dir']) / selected['paths']),
                              manifest['pose_file'], frame_limit, manifest['mimi_file'])
    latents, pose, code_table, codes = clip
    expected = 5 if args.preflight else manifest['expected_latent_frames']
    if latents.shape != (48, expected, *metadata['compatibility']['latent_hw']):
        raise ValueError(f'Unexpected clip geometry: {tuple(latents.shape)}')
    source = AloeChunkSource(manifest['root_dir'], str(Path(manifest['csv']).parent),
                             Path(manifest['csv']).name, bank=manifest['reference_bank'],
                             pose_file=manifest['pose_file'], mimi_file=manifest['mimi_file'])
    refs, keep = source.load_eval_refs(selected['csv_row'], latents[:, :1])
    if int(keep.sum()) != selected['reference_slots_present']:
        raise ValueError('Reference slot coverage changed since input selection')
    provenance['input_sha256'] = {name: digest(item['path']) for name, item in selected['files'].items()}
    provenance['reference_sha256'] = {str(item['property']): digest(item['path']) for item in selected['refs']}
    provenance['reference_keep'] = keep.tolist()
    provenance['picked_reference_sha256'] = hashlib.sha256(refs.contiguous().numpy().tobytes()).hexdigest()
    loader.CKPT_PATH = str(args.vae_path.parent)
    models, resolving = sampler.build_modes(config, Path(sampler.__file__).parent, 'cuda')
    cached = sampler.WavefrontNet(models, latents.shape[2:], 2, 'cuda', True, 1, .2, guidance_scale=1.)
    cached.refs = (refs.transpose(0, 1)[None].cuda(), keep[None].cuda())
    resolver = GraphResolver(resolving, num_steps=1)
    with torch.autocast('cuda', dtype=AUTOCAST_DTYPE, cache_enabled=False):
        resolver.set_refs(refs[None].cuda(), keep[None].cuda())
    poses, code_rows, table = pose[None].cuda(), codes[None].cuda(), code_table.cuda()
    resolved = torch.zeros_like(poses)
    resolved[:, :, 0] = latents[:, 0].cuda()
    times = torch.tensor([.2, 0.], device='cuda')
    audio_cache = {}

    def resolve(frame, committed):
        tokens = resolver_tokens(resolver, [table], code_rows, frame, audio_cache)
        if frame == 1:
            before, rng = committed.clone(), torch.cuda.get_rng_state()
        resolver.frame(resolved, poses, code_rows, frame, committed, times, tokens)
        if frame == 1:
            if not torch.equal(resolver.x, before) or not torch.equal(committed, before):
                raise RuntimeError('One-step resolver must consume the unchanged stabilized endpoint')
            if not torch.equal(torch.cuda.get_rng_state(), rng):
                raise RuntimeError('The direct resolver must not draw new random noise')
            provenance.update(direct_first_input_equal=True, stabilized_input_unchanged=True,
                              resolver_drew_random_noise=False)

    print(json.dumps({'event': 'generating', 'clip': args.clip, 'latent_frames': expected,
                      'reference_keep': keep.tolist()}), flush=True)
    generated = sampler.generate_wavefront(cached, [clip], [selected['seed']], 'cuda', True, on_commit=resolve)[0]
    outputs = [(f'sample_{args.clip}', resolved[0]), (f'sample_{args.clip}_stabilised', generated)]
    provenance['latent_files'] = persist_latents(output, outputs, verify=args.preflight)
    provenance.update(status='generated', generation_seconds=time.monotonic() - started)
    write_json(output / f'generation-complete-{args.clip}.json', provenance)
    if args.generate_only:
        print(json.dumps({'event': 'generation_complete', 'clip': args.clip, 'output': str(output),
                          'elapsed_seconds': provenance['generation_seconds']}), flush=True)
        return
    audio = models.mimi.decode(codes[None, :, :2 * expected].cuda()).audio_values[0, 0].float().cpu().numpy()
    media = outputs if args.decode_stabilised else outputs[:1]
    if args.decode_ground_truth:
        media = [(f'samples_{args.clip}', latents), *media]
    for name, values in media:
        decode_wan22_vae_bounded(models.vae, values[None], output / f'{name}.mp4', 25, audio, 'joint')
    videos = [output / f'{name}.mp4' for name, _ in media]
    if args.decode_ground_truth:
        side_by_side = output / f'sample_{args.clip}_sxs.mp4'
        sampler.side_by_side(output / f'samples_{args.clip}.mp4', output / f'sample_{args.clip}.mp4', side_by_side)
        videos.append(side_by_side)
    provenance.update(status='completed', latent_frames=expected, decoded_frames=4 * expected - 3,
                      video_duration_seconds=(4 * expected - 3) / 25,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      elapsed_seconds=time.monotonic() - started,
                      videos=[{'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)} for path in videos])
    write_json(receipt_path, provenance)
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
