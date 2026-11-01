"""Bounded, isolated speech/pose depth-decoder benchmark on landsend-12 GPU0.

Uses production decoder weights, synthetic aligned tower/link hidden states,
and synthetic valid 32-codebook histories. No trunk, optimizer, training run,
FSDP communication or checkpoint writing is part of these measurements.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import torch


@contextmanager
def deadline(seconds):
    def expired(*_):
        raise TimeoutError(f'Benchmark phase exceeded {seconds} seconds')
    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)


def log(event, **fields):
    print(json.dumps({'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields}), flush=True)


@torch.no_grad()
def load_decoder(M, path, prefix, frozen):
    cfg = M.cfg06()
    with torch.device('meta'):
        decoder = M.Decoder(2560, cfg)
    decoder.to_empty(device='cuda')
    decoder.model.rotary_emb = type(decoder.model.rotary_emb)(config=cfg, device='cuda')
    wanted = {prefix + '.' + key: value for key, value in decoder.state_dict().items()}
    started = time.perf_counter()
    if not (path / 'COMPLETE').is_file():
        raise ValueError(f'Checkpoint is incomplete: {path}')
    if (path / '.metadata').is_file():
        import torch.distributed.checkpoint as dcp
        reader = dcp.FileSystemReader(path)
        metadata = reader.read_metadata().state_dict_metadata
        for key, value in wanted.items():
            if key not in metadata or tuple(metadata[key].size) != tuple(value.shape):
                raise ValueError(f'Checkpoint tensor shape/key mismatch: {key}')
        dcp.load(wanted, storage_reader=reader)
        source = 'DCP subset, full decoder tensors'
    else:
        replica = torch.load(path / 'rep.pt', map_location='cpu', mmap=True, weights_only=True)
        for key, value in wanted.items():
            if key not in replica or tuple(replica[key].shape) != tuple(value.shape):
                raise ValueError(f'Replicated decoder tensor shape/key mismatch: {key}')
            value.copy_(replica[key])
        del replica
        source = 'rep.pt, full replicated decoder tensors'
    if not all(bool(torch.isfinite(value).all()) for value in wanted.values()):
        raise ValueError('Decoder has nonfinite checkpoint values')
    decoder.requires_grad_(not frozen)
    decoder.train(not frozen)
    report = {'checkpoint': str(path), 'prefix': prefix, 'frozen': frozen, 'source': source,
              'parameters': sum(p.numel() for p in decoder.parameters()), 'tensors': len(wanted),
              'seconds': time.perf_counter() - started}
    log('decoder_loaded', **report)
    return decoder, report


def zero_students(models):
    for stream in models.values():
        stream['student'].zero_grad(set_to_none=True)


@torch.no_grad()
def tensor_error(actual, reference_cpu):
    if actual is None or reference_cpu is None:
        return {'pass': actual is None and reference_cpu is None, 'both_none': actual is None and reference_cpu is None}
    diff_sq = torch.zeros((), device='cuda', dtype=torch.float64)
    ref_sq = torch.zeros_like(diff_sq)
    max_abs = torch.zeros((), device='cuda')
    max_ref = torch.zeros_like(max_abs)
    flat = actual.reshape(-1)
    reference_cpu = reference_cpu.reshape(-1)
    finite = torch.ones((), device='cuda', dtype=torch.bool)
    for start in range(0, flat.numel(), 1 << 20):
        value = flat[start:start + (1 << 20)].float()
        ref = reference_cpu[start:start + (1 << 20)].to('cuda', dtype=torch.float32)
        diff = value - ref
        finite &= torch.isfinite(value).all()
        diff_sq += diff.square().sum(dtype=torch.float64)
        ref_sq += ref.square().sum(dtype=torch.float64)
        max_abs = torch.maximum(max_abs, diff.abs().amax())
        max_ref = torch.maximum(max_ref, ref.abs().amax())
    relative_l2 = float((diff_sq / ref_sq.clamp_min(1e-30)).sqrt())
    scaled_max = float(max_abs / max_ref.clamp_min(1e-30))
    # BF16 fusion can change rounding. Report the actual errors as well as a
    # conservative threshold; this is not a bitwise-equivalence claim.
    passed = bool(finite) and relative_l2 <= .02 and scaled_max <= .10
    return {'relative_l2': relative_l2, 'max_absolute': float(max_abs),
            'max_error_over_reference_max': scaled_max, 'finite': bool(finite), 'pass': passed}


def capture_reference(student, outputs):
    dh, dz, metrics = outputs
    return {'hidden_grad': dh.cpu(), 'link_grad': None if dz is None else dz.cpu(),
            'metrics': {key: value.cpu() for key, value in metrics.items()},
            'parameters': {key: None if value.grad is None else value.grad.cpu()
                           for key, value in student.named_parameters()}}


def compare_reference(student, outputs, reference):
    dh, dz, metrics = outputs
    errors = {'hidden_grad': tensor_error(dh, reference['hidden_grad']),
              'link_grad': tensor_error(dz, reference['link_grad'])}
    errors['metrics'] = {key: tensor_error(value, reference['metrics'][key]) for key, value in metrics.items()}
    worst_l2, worst_max, failed, checked = 0., 0., [], 0
    for key, parameter in student.named_parameters():
        error = tensor_error(parameter.grad, reference['parameters'][key])
        worst_l2 = max(worst_l2, error.get('relative_l2', 0.))
        worst_max = max(worst_max, error.get('max_error_over_reference_max', 0.))
        checked += parameter.numel() if parameter.grad is not None else 0
        if not error['pass']:
            failed.append({'parameter': key, **error})
    errors['parameter_gradients'] = {'elements_checked': checked, 'worst_relative_l2': worst_l2,
                                    'worst_max_error_over_reference_max': worst_max,
                                    'failed': failed, 'pass': not failed}
    errors['pass'] = (all(errors[key]['pass'] for key in ('hidden_grad', 'link_grad', 'parameter_gradients'))
                      and all(error['pass'] for error in errors['metrics'].values()))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', nargs='+', type=int, default=[470, 7500])
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--compile-timeout', type=int, default=300)
    parser.add_argument('--skip-compile', action='store_true')
    args = parser.parse_args()
    assert socket.gethostname().split('.')[0] == 'landsend-12', 'Dedicated benchmark host only'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Use only physical GPU0 on landsend-12'
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(71917)
    torch.cuda.manual_seed_all(71917)
    import runtime as R
    from profiling import PhaseProfiler, NullProfile
    R.install_forwards()
    M = R.M
    output = args.output.resolve()
    result = {'benchmark': 'matched_frame_speech_pose_depth', 'host': socket.gethostname(),
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'pid': os.getpid(), 'physical_gpu': 0, 'complete': False,
              'inputs': 'Production decoder weights; deterministic synthetic BF16 tower/link hidden states and valid random codebook histories',
              'frames': args.frames, 'chunk_cap': 470, 'objective_scale': .75 / 3,
              'warmup_iterations': args.warmup, 'timed_iterations': args.iterations,
              'normalization': 'world1/global_frames = 1/local_frames, then 1/31 higher-codebook mean',
              'excludes': ['trunk/towers', 'optimizer', 'FSDP/NCCL communication', 'dataset loading'],
              'timing': 'CUDA events plus CPU dispatch and synchronized wall; CPU and GPU intervals overlap',
              'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), HERE / 'runtime.py', HERE / 'losses.py')},
              'loads': [], 'cases': [], 'errors': []}
    save(output, result)
    models = {'speech': {}, 'pose': {}}
    paths = [
        ('speech', 'student', Path('/home/param/models/imported-weights/ppm-aloe-poseduplex-v5-final/export-498-dcp'), 'dec_s'),
        ('pose', 'student', Path('/home/param/models/imported-weights/ppm-aloe-poseduplex-v5-final/export-498-dcp'), 'dec_p'),
        ('speech', 'teacher', Path('/home/param/models/imported-weights/ppm-aloe-samsep-vox17127-final/export-249-dcp'), 'dec_s'),
        ('pose', 'teacher', Path('/home/param/models/imported-weights/transcript-pose/xabi-original/export-200'), 'dec_p'),
    ]
    for stream, role, path, prefix in paths:
        log('load_start', stream=stream, role=role)
        with deadline(300):
            decoder, report = load_decoder(M, path, prefix, role == 'teacher')
        decoder._profile_name = stream
        models[stream][role] = decoder
        result['loads'].append({'stream': stream, 'role': role, **report})
        save(output, result)
    maximum = max(args.frames)
    hv = torch.randn(maximum, 2560, device='cuda', dtype=torch.bfloat16)
    thv = torch.randn_like(hv)
    zb = torch.randn(maximum, 1024, device='cuda', dtype=torch.bfloat16) * .1
    tzb = torch.randn_like(zb) * .1
    depth = torch.randint(0, 2048, (maximum, 32), device='cuda') + torch.arange(32, device='cuda') * 2048
    light = torch.arange(maximum, device='cuda')
    references, logits_references = {}, {}

    def run(stream, frames):
        model = models[stream]
        return R.decoder_kl(model['student'], model['teacher'], hv[:frames], thv[:frames], depth[:frames], light[:frames],
            zb=zb[:frames] if stream == 'pose' else None, teacher_zb=tzb[:frames] if stream == 'pose' else None,
            global_frames=frames, world=1, dummy=False, chunk_cap=470, step=3, rank=0, seed=71917,
            grad=True, objective_scale=.75 / 3)

    def check_logits(stream, frames, variant):
        # Full 31x2048 output check for one chunk, not top-k or sampled entries.
        model = models[stream]
        checks = {}
        with torch.no_grad():
            for role in ('student', 'teacher'):
                hidden = hv if role == 'student' else thv
                link = zb if role == 'student' else tzb
                logits, labels = model[role](hidden[:frames], depth[:frames], light[:frames],
                                             zb=link[:frames] if stream == 'pose' else None)
                key = stream, role
                if variant == 'baseline':
                    logits_references[key] = logits.cpu()
                else:
                    checks[role] = tensor_error(logits, logits_references[key])
                del logits, labels
        return checks

    original_layer, original_trunk = R.decoder_layer, M._dec_trunk
    variants = ['baseline'] + ([] if args.skip_compile else ['compiled_layers'])
    try:
        for variant in variants:
            if variant == 'compiled_layers':
                log('compile_start', mode='default', dynamic=True, timeout_seconds=args.compile_timeout)
                before = time.perf_counter()
                try:
                    with deadline(args.compile_timeout):
                        compiled_layer = torch.compile(original_layer, mode='default', dynamic=True, fullgraph=False)
                        R.decoder_layer = compiled_layer
                        def compiled_teacher(model, x):
                            cos, sin = model.rotary_emb(x, torch.arange(x.shape[1], device=x.device)[None])
                            h = x
                            for layer in model.layers[:model.config.num_hidden_layers]:
                                h = compiled_layer(layer, h, cos, sin)
                            return model.norm(h)
                        M._dec_trunk = compiled_teacher
                        R.profile = NullProfile()
                        zero_students(models)
                        primed = run('pose', min(args.frames))
                        torch.cuda.synchronize()
                        del primed
                    result['compile'] = {'pass': True, 'mode': 'default', 'dynamic': True,
                                         'seconds_including_first_forward_backward': time.perf_counter() - before,
                                         'student_per_layer_activation_checkpointing': True,
                                         'teacher_and_student_use_compiled_actual_decoder_layer': True}
                    log('compile_complete', **result['compile'])
                except Exception as exc:
                    result['compile'] = {'pass': False, 'error': repr(exc), 'traceback': traceback.format_exc(),
                                         'seconds': time.perf_counter() - before}
                    result['errors'].append(result['compile'])
                    log('compile_failed', error=repr(exc))
                    save(output, result)
                    break
            for stream in ('speech', 'pose'):
                logits_check = check_logits(stream, min(args.frames), variant)
                if any(not error['pass'] for error in logits_check.values()):
                    raise AssertionError(f'Compiled logits differ: {stream}: {logits_check}')
                for frames in args.frames:
                    case = {'variant': variant, 'stream': stream, 'frames': frames,
                            'chunks': math.ceil(frames / 470), 'logits_check': logits_check}
                    result['cases'].append(case)
                    log('case_start', variant=variant, stream=stream, frames=frames)
                    with deadline(120):
                        R.profile = NullProfile()
                        for _ in range(args.warmup):
                            zero_students(models)
                            warmed = run(stream, frames)
                            torch.cuda.synchronize()
                            del warmed
                        zero_students(models)
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        case['initial_allocated_bytes'] = torch.cuda.memory_allocated()
                        timings = []
                        for iteration in range(args.iterations):
                            zero_students(models)
                            profiler = PhaseProfiler(torch.device('cuda', 0))
                            R.profile = profiler
                            profiler.begin(iteration + 1)
                            start, finish = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                            wall = time.perf_counter()
                            start.record()
                            outputs = run(stream, frames)
                            cpu_dispatched = time.perf_counter() - wall
                            finish.record()
                            finish.synchronize()
                            elapsed = time.perf_counter() - wall
                            record = profiler.finish()
                            timings.append({'gpu_seconds': start.elapsed_time(finish) / 1000,
                                            'cpu_dispatch_seconds': cpu_dispatched, 'wall_seconds': elapsed,
                                            'stages': record['stages']})
                            if iteration != args.iterations - 1:
                                del outputs
                        case.update(timings=timings,
                                    median_gpu_seconds=statistics.median(t['gpu_seconds'] for t in timings),
                                    median_cpu_dispatch_seconds=statistics.median(t['cpu_dispatch_seconds'] for t in timings),
                                    median_wall_seconds=statistics.median(t['wall_seconds'] for t in timings),
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    R.profile = NullProfile()
                    key = stream, frames
                    if variant == 'baseline':
                        references[key] = capture_reference(models[stream]['student'], outputs)
                        case['correctness'] = {'baseline_reference': True, 'pass': True}
                    else:
                        case['correctness'] = compare_reference(models[stream]['student'], outputs, references[key])
                        if not case['correctness']['pass']:
                            raise AssertionError(f'Compiled gradient/metric comparison failed: {stream}, {frames}')
                    case['mean_kl'] = float((outputs[2]['kl'] / outputs[2]['count'].clamp_min(1)).mean())
                    case['mean_data_ce'] = float((outputs[2]['ce'] / outputs[2]['count'].clamp_min(1)).mean())
                    case['teacher_gradients_absent'] = all(p.grad is None for p in models[stream]['teacher'].parameters())
                    if not case['teacher_gradients_absent']:
                        raise AssertionError('Frozen teacher received gradients')
                    del outputs
                    zero_students(models)
                    log('case_complete', variant=variant, stream=stream, frames=frames,
                        seconds=case['median_gpu_seconds'], peak_GB=case['peak_allocated_bytes'] / 1e9,
                        correctness=case['correctness']['pass'])
                    save(output, result)
        result['complete'] = True
    except Exception as exc:
        result['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
        log('benchmark_error', error=repr(exc))
        raise
    finally:
        R.decoder_layer, M._dec_trunk, R.profile = original_layer, original_trunk, NullProfile()
        save(output, result)
        zero_students(models)
        log('benchmark_finished', complete=result['complete'], errors=len(result['errors']))


if __name__ == '__main__':
    main()
