"""Fixed-source student AR evaluation and a local, manifest-driven run gallery.

Only the sink and references are visual GT inputs. Stabilizer history is its
generated t=.2 history; resolver history is its own previously resolved output.
RGB metrics are computed before video compression. This module never publishes
to a remote host and never touches an optimizer or distributed collective.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid
import wave

import numpy as np
import torch

from distillation.joint_eval import _evaluation_state, _seed_forward
from distillation.evaluation_identity import evaluation_identity_fields


DRIFT_KEYS = ('background_color_mean', 'background_color_tail',
              'background_texture_mean', 'background_texture_tail')


@contextmanager
def _eager_blocks(net):
    """Keep evaluation-only layouts out of the training compiler cache."""
    previous = getattr(net, 'block_fn', None)
    if previous is None or not hasattr(net, '_block'):
        yield
        return
    try:
        net.block_fn = net._block
        yield
    finally:
        net.block_fn = previous


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tensor(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_numbers(values):
    return [float(v) if math.isfinite(float(v)) else None for v in values]


def summarize_drift(seconds, measurements, *, duration, continuation_start, tail_seconds=5.):
    """Duration-aware AutoBench M1/M2 summaries, preserving signed M2."""
    seconds = np.asarray(seconds, dtype=np.float64)
    if (seconds.ndim != 1 or not np.isfinite(seconds).all() or
            not all(math.isfinite(float(v)) for v in (duration, continuation_start, tail_seconds)) or
            np.any(np.diff(seconds) <= 0) or duration <= 0 or tail_seconds <= 0 or
            not 0 <= continuation_start < duration or np.any(seconds < 0) or np.any(seconds >= duration)):
        raise ValueError('Drift timestamps and continuation interval must be finite and ordered')
    counts = np.asarray(measurements['n_bg'])
    if counts.shape != seconds.shape:
        raise ValueError('Background counts must align with drift timestamps')
    result, masks = {}, {}
    for metric, name in (('m1', 'background_color'), ('m2', 'background_texture')):
        values = np.asarray(measurements[metric], dtype=np.float64)
        if values.shape != seconds.shape:
            raise ValueError('Background measurements must align with drift timestamps')
        valid = np.isfinite(values) & (counts > 0) & (seconds >= continuation_start)
        tail = valid & (seconds >= duration - tail_seconds)
        for suffix, selected in (('mean', valid), ('tail', tail)):
            result[f'{name}_{suffix}'] = float(values[selected].mean()) if selected.any() else None
        masks[metric] = {'valid': valid.tolist(), 'tail_valid': tail.tolist(),
                         'valid_count': int(valid.sum()), 'tail_valid_count': int(tail.sum())}
    return result, masks


@torch.no_grad()
def resolve_generated_sequence(net, rows, stabilized):
    """Direct one-step .2 -> 0 resolution, with generated clean history only."""
    if stabilized.shape != rows['latents'].shape or stabilized.shape[0] != 1:
        raise ValueError('Resolver requires one complete sink-plus-generation sequence')
    if not torch.equal(stabilized[:, :, 0], rows['latents'][:, :, 0]):
        raise ValueError('The supplied sink must be retained exactly')
    resolved = torch.zeros_like(stabilized)
    resolved[:, :, 0] = stabilized[:, :, 0]
    zero = torch.zeros_like(stabilized[:, :, 0])
    plan = [(0, [[('f', 3), ('r', 1, 1), ('r', 2, 2), ('f', 0)]]),
            (1, [[('f', 3), ('f', 1)], [('f', 3), ('f', 2)]]),
            (3, [[('f', 3)]])]
    with net.references(rows['refs'], rows['ref_keep']):
        tokens = net.audio_tokens(rows['audio'])
        for frame in range(1, stabilized.shape[2]):
            x = stabilized[:, :, frame]
            frames = [(x, rows['pose'][:, :, frame], .2, 3, True)]
            for previous in (frame - 2, frame - 1):
                valid = previous >= 1
                frames.append((resolved[:, :, previous] if valid else zero,
                               rows['pose'][:, :, previous] if valid else None, 0., 2, valid))
            frames.append((resolved[:, :, 0], None, 0., 0, True))
            audio = net.frame_audio(tokens, rows['ofi'][:, frame:frame+1], rows['drop'])
            mask = torch.ones((1, 1), dtype=torch.bool, device=x.device)
            velocity, _, _ = net.forward_pass(frames, plan, None, audio, mask, 1)
            resolved[:, :, frame] = x - .2 * velocity[:, :, 0]
    return resolved


@torch.no_grad()
def generate_student_rollout(net, rows, config):
    """Return complete resolved/stabilized sequences; no post-sink GT feedback."""
    from distillation.joint_dmd import generate_stabilizer

    if getattr(net, 'shards', None) is not None:
        raise ValueError('Rank-local rollout evaluation needs an unsharded student')
    latent = rows['latents']
    total = config.get('latent_frames', 189)
    if type(total) is not int or total < 8 or tuple(latent.shape[:1]) != (1,) or latent.shape[2] != total:
        raise ValueError('Rollout rows must contain exactly one complete configured sequence')
    if rows['pose'].shape != latent.shape or rows['ofi'].shape != (1, total):
        raise ValueError('Rollout pose/audio indices must cover the exact video interval')
    if not torch.equal(rows['ofi'][0], torch.arange(total, device=latent.device)):
        raise ValueError('Rollout requires a contiguous source interval starting at its sink')
    seed = config.get('seed', 1729)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Rollout seed must be a nonnegative signed-64-bit integer')
    band_steps = tuple(config.get('band_steps', (2, 2, 2, 2)))
    if len(band_steps) != 4 or len(set(band_steps)) != 1 or any(type(v) is not int or v < 1 for v in band_steps):
        raise ValueError('Rollout requires four equally subdivided stabilizer bands')
    with _evaluation_state(net, latent.device), _eager_blocks(net):
        _seed_forward(seed, latent.device)
        generator = torch.Generator(device=latent.device).manual_seed(seed)
        noise = torch.randn((1, latent.shape[1], total-1, *latent.shape[-2:]),
                            device=latent.device, dtype=torch.float32, generator=generator)
        targets, context = generate_stabilizer(
            net, rows, band_steps, noise=noise, num_targets=total-7, prefix_frames=6,
            history_dropout=(), return_prefix=True)
        stabilized = torch.cat((latent[:, :, :1], context['prefix'], targets), dim=2)
        resolved = resolve_generated_sequence(net, rows, stabilized)
        if not bool(torch.isfinite(resolved).all()) or not bool(torch.isfinite(stabilized).all()):
            raise FloatingPointError('Nonfinite student autoregressive output')
    return resolved, stabilized


class AutoBenchBackgroundScorer:
    """The existing AutoBench segmentation/regions, without ArcFace dependencies."""

    def __init__(self, device, *, segmentation_path=None):
        import kornia
        from transformers import AutoModelForImageSegmentation
        from eval.autobench import ERODE, SEG, SEG_REVISION, Scorer

        self.scorer = Scorer.__new__(Scorer)
        scorer = self.scorer
        scorer.device = device
        scorer.dtype = torch.float16 if torch.device(device).type == 'cuda' else torch.float32
        scorer.batch = 4 if torch.device(device).type == 'cuda' else 1
        kwargs = {} if segmentation_path else {'revision': SEG_REVISION}
        scorer.seg = AutoModelForImageSegmentation.from_pretrained(
            segmentation_path or SEG, trust_remote_code=True, **kwargs).to(device, scorer.dtype).eval()
        scorer.lab, scorer.blur = kornia.color.rgb_to_lab, kornia.filters.gaussian_blur2d
        scorer.mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
        scorer.std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
        radius = torch.arange(-ERODE, ERODE+1, device=device)
        scorer.disk = ((radius[:, None].square() + radius.square()) <= ERODE**2).float()[None, None]

    @torch.no_grad()
    def video(self, frames):
        scorer = self.scorer
        _, background0, lab0, energy0 = [v[0] for v in scorer.regions(frames[:1])]
        output = {'m1': [], 'm2': [], 'n_bg': []}
        for first in range(0, len(frames), scorer.batch):
            for _, background, lab, energy in zip(*scorer.regions(frames[first:first+scorer.batch])):
                common = background & background0
                count = int(common.sum())
                output['n_bg'].append(count)
                if not count:
                    output['m1'].append(float('nan'))
                    output['m2'].append(float('nan'))
                    continue
                output['m1'].append(float((lab[:, common].mean(1)-lab0[:, common].mean(1)).norm()))
                output['m2'].append(float(torch.log(energy[common].mean()/energy0[common].mean())))
        return output


def _decode_video(vae, latents, path, *, fps, audio_np, expected_hw,
                  write_rgb=None, compare_rgb=None):
    """Bounded causal decode, precompression RGB L2, and 1-Hz byte frames."""
    import av
    from inference.bounded_decode import decoded_chunks
    from inference.common import save_with_audio

    path = Path(path)
    temporary = path.with_name(f'.{path.stem}.{uuid.uuid4().hex}.mp4')
    original = path.with_suffix('.mkv')
    original_temporary = original.with_name(f'.{original.stem}.{uuid.uuid4().hex}.mkv')
    count, sample_frames, l2 = 0, [], []
    expected_frames = 4*latents.shape[2]-3
    chunks, container, original_container = decoded_chunks(vae, latents), None, None
    try:
        container = av.open(str(temporary), mode='w', options={'movflags': '+faststart'})
        stream = container.add_stream('libx264', rate=fps)
        stream.height, stream.width = expected_hw
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'medium'}
        original_container = av.open(str(original_temporary), mode='w')
        original_stream = original_container.add_stream('ffv1', rate=fps)
        original_stream.height, original_stream.width = expected_hw
        original_stream.pix_fmt = 'bgr0'
        original_stream.options = {'level': '3'}
        for chunk in chunks:
            rgb = chunk.float().clamp(-1, 1).add(1).div(2).permute(1, 2, 3, 0).cpu().numpy()
            if rgb.shape[1:3] != tuple(expected_hw) or count+len(rgb) > expected_frames:
                raise ValueError('Decoded geometry differs from the fixed rollout definition')
            if not np.isfinite(rgb).all():
                raise FloatingPointError('Nonfinite decoded RGB')
            if write_rgb is not None:
                write_rgb[count:count+len(rgb)] = rgb
            if compare_rgb is not None:
                difference = rgb - compare_rgb[count:count+len(rgb)]
                l2.extend(np.sqrt(np.square(difference, dtype=np.float64).sum(axis=(1, 2, 3))).tolist())
            pixels = (rgb*255).astype(np.uint8)
            for offset, image in enumerate(pixels):
                if (count+offset) % fps == 0:
                    sample_frames.append(image.copy())
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                    container.mux(packet)
                for packet in original_stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                    original_container.mux(packet)
            count += len(rgb)
        if count != expected_frames:
            raise ValueError(f'Decoded {count} frames; expected {expected_frames}')
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        for packet in original_stream.encode():
            original_container.mux(packet)
        original_container.close()
        original_container = None
        save_with_audio(str(temporary), audio_np, tag='rollout-evaluation')
        _mux_lossless_audio(original_temporary, audio_np)
        os.replace(temporary, path)
        os.replace(original_temporary, original)
    finally:
        chunks.close()
        if container is not None:
            container.close()
        if original_container is not None:
            original_container.close()
        temporary.unlink(missing_ok=True)
        original_temporary.unlink(missing_ok=True)
    return {'sample_frames': np.stack(sample_frames), 'l2': l2, 'frames': count, 'original_path': str(original)}


def _mux_lossless_audio(path, audio_np):
    """FFV1 video plus uncompressed PCM audio in Matroska."""
    path = Path(path)
    wav = path.with_name(f'.audio-{uuid.uuid4().hex}.wav')
    muxed = path.with_name(f'.mux-{uuid.uuid4().hex}.mkv')
    try:
        with wave.open(str(wav), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24_000)
            handle.writeframes(np.clip(audio_np*32767, -32768, 32767).astype(np.int16).tobytes())
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(path), '-i', str(wav),
                        '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'pcm_s16le',
                        '-shortest', str(muxed)], check=True, capture_output=True)
        os.replace(muxed, path)
    finally:
        wav.unlink(missing_ok=True)
        muxed.unlink(missing_ok=True)


def _figures(output, timestamps, l2, seconds, measurements, continuation_start):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    l2_path, drift_path = output/'gt_l2_curve.png', output/'background_curves.png'
    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    ax.plot(timestamps, l2, linewidth=1.3)
    ax.axvspan(0, continuation_start, color='gray', alpha=.18, label='Supplied sink + generated startup')
    ax.set(xlabel='Elapsed video time (seconds)', ylabel='RGB Euclidean L2',
           title='Student generation versus VAE-reconstructed GT · RGB [0,1]')
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(l2_path, dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for ax, metric, label in zip(axes, ('m1', 'm2'), ('Background colour drift (CIE76 ΔE)',
                                                  'Background texture drift (signed log ratio)')):
        for source, name in (('generated', 'Student generation'), ('gt', 'VAE-reconstructed GT')):
            ax.plot(seconds, measurements[source][metric], marker='.', label=name)
        ax.axvspan(0, continuation_start, color='gray', alpha=.18)
        ax.axhline(0, color='gray', linewidth=.6)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend()
    axes[-1].set_xlabel('Elapsed video time (seconds)')
    fig.savefig(drift_path, dpi=160)
    plt.close(fig)
    return l2_path, drift_path


def evaluate_rollout(net, rows, config, output_dir, *, vae=None, scorer=None, audio_np=None):
    """Create one complete local evaluation; return a strict-JSON artifact manifest.

    ``config['rollout_eval']`` requires run_id, a checkpoint or explicit in-memory
    evaluation identity, weights ('raw' or 'ema'), vae_path and vae_sha256.
    Defaults are one fixed 189-latent clip, seed
    1729, 25 fps and RGB 256x480. ``rows`` is batch-one eval_batch(num_targets=182)
    after normal preparation. ``audio_np`` must be matching mono 24-kHz audio.
    For tests, injected VAE/scorer avoid loading models; supplied VAE identity is
    still recorded from the explicitly configured checkpoint.
    """
    # Decoder/segmentation constructors can consume RNG as well as generation.
    with _evaluation_state(net, rows['latents'].device), _eager_blocks(net):
        return _evaluate_rollout(net, rows, config, output_dir, vae=vae,
                                 scorer=scorer, audio_np=audio_np)


def _evaluation_geometry(rows, options):
    height, width = options.get('height', 480), options.get('width', 256)
    if (height, width) != (480, 256):
        raise ValueError('The versioned RGB L2 metric requires width 256 and height 480')
    decoded_hw = tuple(int(size) * 16 for size in rows['latents'].shape[-2:])
    if decoded_hw != (height, width):
        raise ValueError(f'Wan2.2 latent geometry decodes to {decoded_hw}, not {(height, width)}')
    return height, width


def _evaluate_rollout(net, rows, config, output_dir, *, vae=None, scorer=None, audio_np=None):
    options = dict(config.get('rollout_eval', config))
    options.setdefault('band_steps', config.get('band_steps', (2, 2, 2, 2)))
    identity_fields = evaluation_identity_fields(options)
    for key in ('run_id', 'weights', 'vae_path', 'vae_sha256'):
        if not options.get(key):
            raise ValueError(f'Rollout evaluation requires explicit {key}')
    if options['weights'] not in ('raw', 'ema'):
        raise ValueError('Rollout weights must be raw or ema')
    fps, total = options.get('fps', 25), options.get('latent_frames', 189)
    if fps != 25 or total != 189:
        raise ValueError('The versioned 30-second metric requires 189 latents at 25 fps')
    height, width = _evaluation_geometry(rows, options)
    decoded_frames = 4*total-3
    duration, continuation_start = decoded_frames/fps, (1+4*6)/fps
    if audio_np is None:
        raise ValueError('Matching decoded 24-kHz source audio is required')
    audio_np = np.asarray(audio_np, dtype=np.float32)
    audio_length = math.ceil(duration*24_000)
    if audio_np.ndim != 1 or len(audio_np) < audio_length or not np.isfinite(audio_np).all():
        raise ValueError('Source audio must be finite mono 24-kHz and cover the full video')
    audio_np = audio_np[:audio_length]
    audio_sha256 = hashlib.sha256(audio_np.tobytes()).hexdigest()
    if _hash_file(options['vae_path']) != options['vae_sha256']:
        raise ValueError('Wan2.2 VAE checkpoint hash differs from the frozen evaluation identity')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'rollout.json').exists():
        existing = json.loads((output/'rollout.json').read_text())
        expected = {'run_id': str(options['run_id']), **identity_fields,
                    'weights': options['weights'], 'seed': options.get('seed', 1729),
                    'vae_sha256': options['vae_sha256'], 'audio_sha256': audio_sha256,
                    'band_steps': list(options['band_steps']), 'latent_frames': total, 'fps': fps,
                    'source': options.get('source', {})}
        if existing.get('status') != 'complete' or any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError('Completed rollout has a different evaluation identity')
        for key, digest in existing['input_sha256'].items():
            if _hash_tensor(rows[key]) != digest:
                raise ValueError('Completed rollout inputs differ from this evaluation')
        for artifact in existing['artifacts'].values():
            if _hash_file(artifact['path']) != artifact['sha256']:
                raise ValueError('Completed rollout artifact hash verification failed')
        existing['manifest_path'] = str((output/'rollout.json').resolve())
        return existing
    generated, stabilized = generate_student_rollout(net, rows, options)
    torch.save({'generated': generated.cpu(), 'stabilized': stabilized.cpu(),
                'gt': rows['latents'].detach().cpu()}, output/'latents.pt')
    if vae is None:
        from models.autoencoders.wanvae import Wan2_2_VAE
        vae = Wan2_2_VAE(vae_pth=options['vae_path'], dtype=torch.float32, device=rows['latents'].device)
    scratch = output/f'.gt-rgb-{uuid.uuid4().hex}.npy'
    reference = np.lib.format.open_memmap(scratch, mode='w+', dtype=np.float32,
                                          shape=(decoded_frames, height, width, 3))
    try:
        gt = _decode_video(vae, rows['latents'], output/'gt_reconstructed.mp4', fps=fps,
                           audio_np=audio_np, expected_hw=(height, width), write_rgb=reference)
        reference.flush()
        result = _decode_video(vae, generated, output/'generated.mp4', fps=fps,
                               audio_np=audio_np, expected_hw=(height, width), compare_rgb=reference)
    finally:
        del reference
        scratch.unlink(missing_ok=True)
    del vae, generated, stabilized
    if scorer is None:
        scorer = AutoBenchBackgroundScorer(rows['latents'].device,
                                           segmentation_path=options.get('segmentation_path'))
    measurements = {'generated': scorer.video(result['sample_frames']), 'gt': scorer.video(gt['sample_frames'])}
    seconds = np.arange(len(result['sample_frames']), dtype=np.float64)
    scalars, validity = summarize_drift(seconds, measurements['generated'], duration=duration,
                                        continuation_start=continuation_start)
    _, gt_validity = summarize_drift(seconds, measurements['gt'], duration=duration,
                                     continuation_start=continuation_start)
    timestamps = np.arange(decoded_frames, dtype=np.float64)/fps
    l2_path, drift_path = _figures(output, timestamps, result['l2'], seconds, measurements, continuation_start)
    curves = {'format': 'student-ar-rgb-drift-30s-v1', 'timestamps_seconds': timestamps.tolist(),
              'rgb_l2': result['l2'], 'drift_seconds': seconds.tolist(),
              'generated': {key: _json_numbers(values) for key, values in measurements['generated'].items()},
              'gt': {key: _json_numbers(values) for key, values in measurements['gt'].items()},
              'validity': validity, 'gt_validity': gt_validity,
              'continuation_start_seconds': continuation_start, 'tail_start_seconds': duration-5}
    _atomic_json(output/'curves.json', curves)
    paths = {'generated_video': output/'generated.mp4', 'gt_video': output/'gt_reconstructed.mp4',
             'original_video': output/'generated.mkv', 'original_gt_video': output/'gt_reconstructed.mkv',
             'gt_l2_curve': l2_path, 'background_curves': drift_path,
             'curves': output/'curves.json', 'latents': output/'latents.pt'}
    artifacts = {key: {'path': str(path.resolve()), 'sha256': _hash_file(path), 'bytes': path.stat().st_size}
                 for key, path in paths.items()}
    manifest = {'format': 'student-ar-evaluation-30s-v1', 'status': 'complete',
                'run_id': str(options['run_id']), **identity_fields,
                'weights': options['weights'], 'seed': options.get('seed', 1729),
                'source': options.get('source', {}),
                'input_sha256': {key: _hash_tensor(rows[key]) for key in
                                 ('latents', 'pose', 'ofi', 'audio', 'refs', 'ref_keep', 'drop')},
                'vae_path': str(options['vae_path']), 'vae_sha256': options['vae_sha256'],
                'audio_sha256': audio_sha256,
                'autobench': {'segmentation_model': 'ZhengPeng7/BiRefNet-portrait',
                              'segmentation_revision': 'b6561965a70070d9143fd9e558f6ca3c481510db',
                              'segmentation_path': options.get('segmentation_path'),
                              'background': 'eroded background intersected with this video own sink background',
                              'erode_pixels': 7, 'gaussian_sigma_pixels': 1.5, 'sample_hz': 1},
                'gt_definition': 'VAE-reconstructed GT', 'rgb_range': [0, 1],
                'rgb_l2_definition': 'sqrt(sum_CHW((generated_RGB - reconstructed_GT_RGB)^2))',
                'metric_source': 'decoded RGB before any lossy video encoding; AutoBench receives uint8 RGB at 1 Hz',
                'video_encoding': {'codec': 'h264', 'crf': 18, 'pixel_format': 'yuv420p', 'audio': 'AAC 128kbps'},
                'original_encoding': {'codec': 'ffv1', 'pixel_format': 'bgr0', 'audio': 'PCM16 24kHz',
                                      'rgb_quantization': 'uint8 floor(255*clipped_float_RGB); metrics use float RGB'},
                'latent_frames': total, 'video_frames': decoded_frames, 'fps': fps,
                'duration_seconds': duration, 'startup_generated_latents': 6,
                'continuation_start_seconds': continuation_start,
                'band_steps': list(options['band_steps']), 'resolver_steps': 1,
                'resolver_input': 'direct stabilizer t=.2 endpoint; no additional noise',
                'history': 'generated stabilizer endpoints and generated resolver outputs; no GT feedback',
                'scalars': scalars, 'artifacts': artifacts}
    _atomic_json(output/'rollout.json', manifest)
    manifest['manifest_path'] = str((output/'rollout.json').resolve())
    return manifest


_GALLERY_HTML = '''<!doctype html><meta charset="utf-8"><title>Training evaluations</title>
<style>body{font:16px system-ui;background:#111;color:#eee;margin:30px}select{font:inherit}main{max-width:1100px;margin:auto}.videos{display:flex;gap:16px}.videos figure{margin:16px 0;width:50%}video,img{width:100%}figcaption{margin:8px 0}a{color:#8cf}</style>
<main><h1 id="title">Training evaluations</h1><label>Evaluation <select id="round"></select></label>
<p id="note"></p><div class="videos"><figure><figcaption>Student generation</figcaption><video id="gen" controls></video></figure><figure><figcaption>VAE-reconstructed GT</figcaption><video id="gt" controls muted></video></figure></div><p><a id="original">Download lossless generation</a> · <a id="originalgt">Download lossless GT reconstruction</a></p><img id="l2"><img id="drift"><div id="sensitivity"></div><p><a id="manifest">Evaluation manifest</a></p></main>
<script>let videos=[document.getElementById('gen'),document.getElementById('gt')];for(let source of videos){let other=videos.find(v=>v!==source);source.addEventListener('play',()=>{if(other.paused)other.play().catch(()=>{})});source.addEventListener('pause',()=>{if(!other.paused)other.pause()});source.addEventListener('seeking',()=>{if(Math.abs(other.currentTime-source.currentTime)>.04)other.currentTime=source.currentTime});source.addEventListener('ratechange',()=>{if(other.playbackRate!==source.playbackRate)other.playbackRate=source.playbackRate});source.addEventListener('timeupdate',()=>{if(!source.paused&&Math.abs(other.currentTime-source.currentTime)>.1)other.currentTime=source.currentTime})}fetch('evaluations.json').then(r=>r.json()).then(data=>{document.getElementById('title').textContent=data.run_id;let select=document.getElementById('round');for(let e of data.entries){let o=document.createElement('option');o.value=e.id;o.textContent=(e.evaluation_kind==='in_memory'?'Student '+e.student_updates+' · in-memory':e.checkpoint_id)+' · '+e.weights;select.appendChild(o)}function show(){let e=data.entries.find(e=>e.id===select.value);for(let [id,key]of [['gen','generated_video'],['gt','gt_video'],['l2','gt_l2_curve'],['drift','background_curves']])document.getElementById(id).src=e.artifacts[key];for(let [id,key]of [['original','original_video'],['originalgt','original_gt_video']]){let a=document.getElementById(id);a.href=e.artifacts[key]||'';a.hidden=!e.artifacts[key]}let s=document.getElementById('sensitivity');s.replaceChildren();for(let [key,path]of Object.entries(e.artifacts)){if((key.startsWith('sensitivity')||['stabilizer_context_gain','resolver_context_gain'].includes(key))&&path.endsWith('.png')){let f=document.createElement('figure'),i=document.createElement('img'),c=document.createElement('figcaption');i.src=path;c.textContent=e.captions[key]||key;f.append(c,i);s.appendChild(f)}}document.getElementById('manifest').href=e.manifest;document.getElementById('note').textContent='30.12 seconds · first second is supplied sink and generated startup · '+e.weights+' weights'}select.onchange=show;select.selectedIndex=select.options.length-1;if(data.entries.length)show()});</script>'''


def append_local_gallery(manifest, gallery_dir):
    """Atomically append a complete, hash-verified local entry; no remote publish."""
    if manifest.get('status') != 'complete':
        raise ValueError('Only completed evaluation entries may enter the gallery')
    root = Path(gallery_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity_fields = evaluation_identity_fields(manifest)
    identity = f"{identity_fields.get('evaluation_id') or identity_fields['checkpoint_id']}-{manifest['weights']}"
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', identity):
        raise ValueError('Evaluation gallery identity contains unsafe path characters')
    with (root/'.manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = root/'evaluations.json'
        index = json.loads(index_path.read_text()) if index_path.exists() else {
            'format': 'training-evaluation-gallery-v1', 'run_id': manifest['run_id'], 'entries': []}
        if index['run_id'] != manifest['run_id']:
            raise ValueError('A training run must have its own gallery')
        final = root/'media'/identity
        existing = next((entry for entry in index['entries'] if entry['id'] == identity), None)
        if final.exists():
            # A previous attempt may have committed immutable assets just before
            # interruption. Validate their identity and hashes before indexing.
            local_manifest = json.loads((final/'rollout.json').read_text())
            for key in ('run_id', *identity_fields, 'weights', 'seed', 'input_sha256', 'vae_sha256'):
                if local_manifest.get(key) != manifest.get(key):
                    raise ValueError('Existing gallery entry has a different evaluation identity')
            if set(local_manifest['artifacts']) != set(manifest['artifacts']):
                raise ValueError('Existing gallery entry has a different artifact set')
            relative = {}
            for key, artifact in manifest['artifacts'].items():
                saved = local_manifest['artifacts'][key]
                if saved['sha256'] != artifact['sha256'] or _hash_file(final/saved['path']) != artifact['sha256']:
                    raise ValueError('Existing gallery artifact differs from the completed evaluation')
                relative[key] = f"media/{identity}/{saved['path']}"
            if existing is not None:
                if existing['artifacts'] != relative:
                    raise ValueError('Gallery index differs from its verified immutable media')
                return {'index_path': str((root/'index.html').resolve()), 'manifest_path': str(index_path.resolve()),
                        'entry_id': identity, 'publication_status': 'local_only_requires_host_upload_and_browser_verification'}
            index['entries'].append({'id': identity, **identity_fields,
                                     'weights': manifest['weights'], 'artifacts': relative,
                                     'captions': manifest.get('captions', {}),
                                     'manifest': f'media/{identity}/rollout.json'})
            (root/'index.html').write_text(_GALLERY_HTML)
            _atomic_json(index_path, index)
            return {'index_path': str((root/'index.html').resolve()), 'manifest_path': str(index_path.resolve()),
                    'entry_id': identity, 'publication_status': 'local_only_requires_host_upload_and_browser_verification'}
        if existing is not None:
            raise ValueError('Gallery index references missing immutable assets')
        staging = root/'media'/f'.{identity}.{uuid.uuid4().hex}.tmp'
        staging.mkdir(parents=True)
        try:
            relative = {}
            gallery_manifest = dict(manifest, artifacts={})
            for key, artifact in manifest['artifacts'].items():
                if not re.fullmatch(r'[A-Za-z0-9_.-]+', key):
                    raise ValueError('Gallery artifact key contains unsafe path characters')
                source = Path(artifact['path'])
                destination = staging/f'{key}{source.suffix}'
                shutil.copy2(source, destination)
                if _hash_file(destination) != artifact['sha256']:
                    raise ValueError('Gallery media hash verification failed')
                relative[key] = f'media/{identity}/{destination.name}'
                gallery_manifest['artifacts'][key] = dict(artifact, path=destination.name,
                                                          source_path=str(source))
            _atomic_json(staging/'rollout.json', gallery_manifest)
            os.replace(staging, final)
            index['entries'].append({'id': identity, **identity_fields,
                                     'weights': manifest['weights'], 'artifacts': relative,
                                     'captions': manifest.get('captions', {}),
                                     'manifest': f'media/{identity}/rollout.json'})
            (root/'index.html').write_text(_GALLERY_HTML)
            _atomic_json(index_path, index)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return {'index_path': str((root/'index.html').resolve()), 'manifest_path': str(index_path.resolve()),
            'entry_id': identity, 'publication_status': 'local_only_requires_host_upload_and_browser_verification'}
