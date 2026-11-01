"""Single-GPU numerical checks for teacher-partner streaming audio.

Loads the real frozen Mimi and Voxtral models, never a full PPM. Results include
FP32 decoding errors and BF16 frontend differences rather than assuming that
fused kernels are bit-identical to the full-sequence encoder.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .partner_audio import AudioTimeline, StreamingDecoder, StreamingFront
from .streaming import vox_stream as VS


def differences(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f'Comparison shapes differ: {tuple(actual.shape)} vs {tuple(expected.shape)}')
    a, b = actual.detach().float(), expected.detach().float()
    delta = a - b
    af, bf = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    cosine = torch.nn.functional.cosine_similarity(af, bf, dim=-1)
    return {'shape': list(a.shape), 'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            'bitwise_equal': bool(torch.equal(actual, expected)),
            'max_absolute': float(delta.abs().max()), 'mean_absolute': float(delta.abs().mean()),
            'rms_error': float(delta.square().mean().sqrt()),
            'relative_l2': float(delta.norm() / b.norm().clamp_min(1e-20)),
            'minimum_cosine': float(cosine.min()), 'mean_cosine': float(cosine.mean())}


class _StackAdapter(nn.Module):
    """Observer only: retain all four encoder vectors at the actor input slot."""
    def forward(self, value):
        return value.reshape(value.shape[0], -1, 4 * value.shape[-1])


def fake_engine(batch, frames, device):
    prefixes = 5 + torch.arange(batch, device=device) * 4
    return SimpleNamespace(d0=device, NI=batch,
        m=SimpleNamespace(vadapter=_StackAdapter().to(device), vgain=nn.Identity().to(device)),
        Pv={device: prefixes},
        _opd_audio=torch.zeros(batch, int(prefixes.max()) + frames + 16, 4 * 1280,
                              device=device, dtype=torch.bfloat16))


@torch.no_grad()
def check_paged_frontend(front, native, fa, args, device, emit):
    """Cross the1024-position ring boundary, then rewind across that boundary."""
    frames, known, first_stop, rewind = args.paged_frames, 254, 260, 253
    rng = np.random.default_rng(args.seed + 17)
    raw = [rng.normal(0, .03, known * VS.SPF24).astype(np.float32) for _ in range(args.batch)]
    replacement = torch.as_tensor(rng.normal(0, .03, (args.batch, VS.SPF24)).astype(np.float32), device=device)
    streams, timelines = {}, {}
    for mode in ('contiguous', 'paged'):
        timeline = AudioTimeline(raw, frames, device, inverse_gains=[1.] * args.batch)
        engine = fake_engine(args.batch, frames, device)
        # Even this short check reserves the complete production logical
        # timeline. The paged cache must stay physically bounded at1024.
        engine.n_fr = 1500
        streams[mode] = StreamingFront(front, engine, timeline, fa=fa, fused_ops=native.FO,
            frames=frames, rewind=8, graphs=True, cache_mode=mode)
        timelines[mode] = timeline
    def compare(stop):
        actual = streams['paged'].latents[:, :4 * stop]
        expected = streams['contiguous'].latents[:, :4 * stop]
        stats = differences(actual, expected)
        if not stats['finite'] or stats['minimum_cosine'] < args.paged_min_cosine:
            raise AssertionError(f'Paged/contiguous frontend mismatch: {stats}')
        return stats
    for frame in range(first_stop):
        for stream in streams.values():
            stream.step(frame, speculative=True)
    initial = compare(first_stop)
    emit('frontend_paged_first_wrap', **initial)
    for mode, stream in streams.items():
        timelines[mode].append(known, replacement)
        stream.rewind(rewind)
        # Encoder positions1012..1035 straddle the1024-position physical wrap.
        stream.step_block(rewind, 6, speculative=True)
        stream.step(first_stop - 1, speculative=True)
    corrected = compare(first_stop)
    for frame in range(first_stop, frames):
        for stream in streams.values():
            stream.step(frame, speculative=True)
    final = compare(frames)
    report = {'frames': frames, 'changed_audio_frame': known, 'rewind_frame': rewind,
        'rewind_block_encoder_positions': [rewind * 4, (rewind + 6) * 4],
        'before_rewind': initial, 'after_rewind': corrected, 'final': final,
        'minimum_cosine_required': args.paged_min_cosine,
        'caches': {mode: stream.cache_report() for mode, stream in streams.items()}}
    emit('frontend_paged_comparison', **report)
    return report


@torch.no_grad()
def check_mimi(args, device, emit):
    from transformers import MimiModel
    started = time.monotonic()
    model = MimiModel.from_pretrained(args.mimi, local_files_only=True,
        attn_implementation='sdpa', torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
    emit('mimi_loaded', seconds=time.monotonic() - started)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    codes = torch.randint(0, 2048, (args.batch, 32, max(args.frames)), generator=generator, device=device)
    offsets = torch.arange(32, device=device) * 2048
    cases = []
    for frames in args.frames:
        case_started = time.monotonic()
        reference = model.decode(codes[:, :, :frames]).audio_values[:, 0]
        per_backend = {}
        for graphs in (False, True):
            stream = StreamingDecoder(model, args.batch, device, graphs=graphs)
            chunks = []
            for frame in range(frames):
                chunks.append(stream.decode(frame, codes[:, :, frame] + offsets).clone())
            actual = torch.cat(chunks, dim=1)
            stats = differences(actual, reference)
            split = min(8, frames - 1) * VS.SPF24
            stats['warm_prefix'] = differences(actual[:, :split], reference[:, :split])
            stats['appended_suffix'] = differences(actual[:, split:], reference[:, split:])
            torch.testing.assert_close(actual, reference, atol=args.mimi_atol, rtol=args.mimi_rtol)
            per_backend['graph' if graphs else 'eager'] = stats
            del stream, chunks, actual
        cases.append({'frames': frames, 'seconds': time.monotonic() - case_started, **per_backend})
        emit('mimi_case', **cases[-1])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {'cases': cases, 'atol': args.mimi_atol, 'rtol': args.mimi_rtol,
            'seconds': time.monotonic() - started}


@torch.no_grad()
def check_frontend(args, device, emit):
    from . import runtime
    native = runtime.load_native()
    import voxtral_front
    fa = native.fa3_op._mod().flash_attn_interface
    started = time.monotonic()
    front = voxtral_front.VoxtralFront(path=args.voxtral, device=device,
                                      attn_implementation='sdpa')
    emit('voxtral_loaded', seconds=time.monotonic() - started)
    frames, known, changed, rewind = 14, 8, 8, 7
    rng = np.random.default_rng(args.seed + 1)
    raw = [rng.normal(0, .03, known * VS.SPF24).astype(np.float32) for _ in range(args.batch)]
    replacement = rng.normal(0, .03, (args.batch, VS.SPF24)).astype(np.float32)
    timeline = AudioTimeline(raw, frames, device, inverse_gains=[1.] * args.batch)
    engine = fake_engine(args.batch, frames, device)
    stream = StreamingFront(front, engine, timeline, fa=fa, fused_ops=native.FO,
                            frames=frames, rewind=8, graphs=True)
    for frame in range(frames):
        stream.step(frame, speculative=True)
    before = stream.latents[:, :4 * frames].clone()
    timeline.append(changed, torch.as_tensor(replacement, device=device))
    stream.rewind(rewind)
    for frame in range(rewind, frames):
        stream.step(frame, speculative=True)
    corrected = stream.latents[:, :4 * frames].clone()
    unchanged = differences(corrected[:, :4 * rewind], before[:, :4 * rewind])
    if not unchanged['bitwise_equal']:
        raise AssertionError('Frontend rewind changed a committed earlier feature')
    mutation = differences(corrected[:, 4 * rewind:], before[:, 4 * rewind:])
    if mutation['bitwise_equal']:
        raise AssertionError('Replacing speculative silence failed to affect later frontend features')

    fresh_prefixes = [np.concatenate((raw[i], replacement[i])) for i in range(args.batch)]
    fresh_timeline = AudioTimeline(fresh_prefixes, frames, device, inverse_gains=[1.] * args.batch)
    fresh_engine = fake_engine(args.batch, frames, device)
    fresh = StreamingFront(front, fresh_engine, fresh_timeline, fa=fa, fused_ops=native.FO,
                           frames=frames, rewind=8, graphs=True)
    for frame in range(frames):
        fresh.step(frame, speculative=True)
    fresh_latents = fresh.latents[:, :4 * frames]
    replay = differences(corrected, fresh_latents)
    torch.testing.assert_close(corrected, fresh_latents, atol=0, rtol=0)
    injection = []
    for row in range(args.batch):
        prefix = int(engine.Pv[device][row])
        observed = engine._opd_audio[row, prefix:prefix + frames]
        expected = corrected[row].reshape(frames, 4 * 1280)
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        injection.append(differences(observed, expected))
    emit('frontend_rewind', unchanged_prefix=unchanged, corrected_vs_fresh=replay,
         correction_effect=mutation, adapter_slot_checks=injection)

    # One extra physical frame keeps all measured frames away from the native
    # full encoder's right-edge reflection/resampling convention. The known
    # prefix and replacement are followed by the same digital silence used by
    # the speculative streaming input, including the 72-sample lookahead.
    full_wave = fresh_timeline.buffer[:, VS.CTX24:VS.CTX24 + (frames + 1) * VS.SPF24]
    native_latents = front.encode(full_wave, frames + 1)[:, :4 * frames]
    native_stats = differences(fresh_latents, native_latents)
    if not native_stats['finite'] or native_stats['minimum_cosine'] < args.front_min_cosine:
        raise AssertionError(f'Fused/native frontend agreement is too low: {native_stats}')
    emit('frontend_native_comparison', **native_stats)
    block_report = None
    if args.check_block:
        block_engine = fake_engine(args.batch, frames, device)
        block = StreamingFront(front, block_engine, fresh_timeline, fa=fa, fused_ops=native.FO,
                               frames=frames, rewind=8, graphs=True)
        for frame in range(7):
            block.step(frame, speculative=True)
        block.step_block(7, 6, speculative=True)
        block.step(13, speculative=True)
        block_latents = block.latents[:, :4 * frames].clone()
        block_vs_sequential = differences(block_latents, fresh_latents)
        if block_vs_sequential['minimum_cosine'] < args.front_min_cosine:
            raise AssertionError(f'Block/sequential frontend agreement is too low: {block_vs_sequential}')
        block.rewind(7)
        block.step_block(7, 6, speculative=True)
        block.step(13, speculative=True)
        graph_replay = differences(block.latents[:, :4 * frames], block_latents)
        torch.testing.assert_close(block.latents[:, :4 * frames], block_latents, atol=0, rtol=0)
        block.rewind(11)
        for frame in range(11, frames):
            block.step(frame, speculative=True)
        interior_replay = differences(block.latents[:, :4 * frames], fresh_latents)
        torch.testing.assert_close(block.latents[:, :4 * 11], block_latents[:, :4 * 11], atol=0, rtol=0)
        if interior_replay['minimum_cosine'] < args.front_min_cosine:
            raise AssertionError(f'Internal block rewind lost convolution state: {interior_replay}')
        timings = {}
        for label, target, use_block in (('sequential', fresh, False), ('block', block, True)):
            events = []
            for _ in range(8):
                target.rewind(7)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                if use_block:
                    target.step_block(7, 6, speculative=True)
                else:
                    for frame in range(7, 13):
                        target.step(frame, speculative=True)
                end.record()
                events.append((begin, end))
            torch.cuda.synchronize(device)
            timings[label + '_six_frame_ms'] = float(np.median([a.elapsed_time(b) for a, b in events]))
        timings['speedup'] = timings['sequential_six_frame_ms'] / timings['block_six_frame_ms']
        block_report = {'count': 6, 'encoder_positions': 24, 'block_vs_sequential': block_vs_sequential,
                        'cached_graph_replay': graph_replay, 'interior_rewind_from_frame': 11,
                        'interior_rewind_vs_sequential': interior_replay, 'timings': timings}
        emit('frontend_block_comparison', **block_report)
    paged_report = check_paged_frontend(front, native, fa, args, device, emit) if args.check_paged else None
    return {'frames': frames, 'real_prefill_frames': known, 'changed_physical_frame': changed,
            'rewind_from_frame': rewind, 'lookahead_samples_24khz': VS.LOOK24,
            'rewind_vs_fresh_graph': replay, 'unchanged_prefix': unchanged,
            'correction_effect': mutation, 'native_full_encoder': native_stats,
            'native_full_frames_encoded': frames + 1, 'native_frames_compared': frames,
            'native_bitwise_agreement_required': False, 'minimum_cosine_required': args.front_min_cosine,
            'block': block_report,
            'paged': paged_report,
            'adapter': 'identity stack observer; no trained PPM adapter loaded',
            'seconds': time.monotonic() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mimi', default='/home/param/models/mimi')
    parser.add_argument('--voxtral', default='/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--frames', default='16,32')
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--mimi-atol', type=float, default=5e-5)
    parser.add_argument('--mimi-rtol', type=float, default=1e-4)
    parser.add_argument('--front-min-cosine', type=float, default=.99)
    parser.add_argument('--check-block', action='store_true',
                        help='Also verify/measure the six-frame frontend graph and an internal rewind boundary')
    parser.add_argument('--check-paged', action='store_true',
                        help='Compare paged vs contiguous KV through a physical wrap and a rewind crossing it')
    parser.add_argument('--paged-frames', type=int, default=300)
    parser.add_argument('--paged-min-cosine', type=float, default=.9995)
    args = parser.parse_args(argv)
    args.frames = [int(value) for value in args.frames.split(',')]
    if args.batch < 1 or not args.frames or min(args.frames) < 2 or args.max_seconds < 1:
        parser.error('Require positive batch/deadline and at least two frames per Mimi case')
    if args.paged_frames < 260:
        parser.error('--paged-frames must include at least260 physical frames to exercise a KV wrap')
    device = torch.device(args.device)
    if device.type != 'cuda':
        parser.error('This parity harness requires one CUDA GPU')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'device': str(device), 'gpu': torch.cuda.get_device_name(device),
              'batch': args.batch, 'seed': args.seed, 'mimi_path': args.mimi, 'voxtral_path': args.voxtral,
              'stages': [], 'scope': 'real streaming audio models only; no full PPM loaded'}
    def emit(event, **fields):
        row = {'event': event, 'elapsed_seconds': time.monotonic() - started, **fields}
        report['stages'].append(row)
        temporary = args.report.with_suffix(args.report.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.report)
        print(json.dumps(row, allow_nan=False), flush=True)
    def expired(*_):
        raise TimeoutError(f'Audio parity check exceeded {args.max_seconds} seconds')
    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.max_seconds)
    try:
        emit('start')
        report['mimi'] = check_mimi(args, device, emit)
        report['frontend'] = check_frontend(args, device, emit)
        torch.cuda.synchronize(device)
        report['status'] = 'passed'
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        emit('passed', seconds=time.monotonic() - started)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        emit('failed', error=report['error'])
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    main()
