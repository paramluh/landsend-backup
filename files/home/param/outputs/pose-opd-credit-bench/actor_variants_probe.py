#!/usr/bin/env python3
"""Standalone, unsaved 750-frame actor experiment; never a training entrypoint.

One immutable BF16 checkpoint remains resident on two GPUs. Each candidate gets
fresh caches/graphs and identical real data, frontend features, and RNG seeds.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=750)
    parser.add_argument('--capacity-frames', type=int)
    parser.add_argument('--instances', type=int, default=8)
    parser.add_argument('--cut', type=int, default=30)
    parser.add_argument('--variants', default='baseline,linear_down,baseline,linear_all,baseline')
    parser.add_argument('--profile-stages', action='store_true')
    parser.add_argument('--tune-gemv', action='store_true')
    parser.add_argument('--probe-depth', action='store_true', help='After timing, compare cached and replay depth on identical final-frame roots')
    parser.add_argument('--qwen32-config', default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B')
    parser.add_argument('--qwen4-config', default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-4B')
    parser.add_argument('--voxtral', default='/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import triton
    from finetune.pose_opd_credit.config import TrainingConfig
    from finetune.pose_opd_credit.data import ProductionRows
    from finetune.pose_opd_credit.diagnostics import benchmark_prefix
    from finetune.pose_opd_credit.actor import build_actor, release_publication_cache
    from finetune.pose_opd_credit.rollout import make_fastppm_class, rollout_rows, audio_features_from_latents
    from finetune.pose_opd_credit.train import checkpoint_spec, load_actor_native, make_plan, source_digest
    from finetune.pose_opd_credit import runtime
    from finetune.duplex_distill.checkpoints import load_checkpoint

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 2:
        raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly the isolated two-GPU pair')
    if args.instances != 8 or args.frames <= 27:
        raise ValueError('This bounded experiment requires8 instances and more than27 frames')
    if os.environ.get('PPM_AUTOTUNE', '0') != '0':
        raise ValueError('The baseline requires fixed native GEMV settings: PPM_AUTOTUNE=0')
    variants = args.variants.split(',')
    if any(value not in ('baseline', 'linear_down', 'linear_all', 'tuned_down', 'tuned_all') for value in variants):
        raise ValueError('Unknown variant')
    if any(value.startswith('tuned') for value in variants) and not args.tune_gemv:
        raise ValueError('Tuned variants require --tune-gemv')
    devices = [torch.device('cuda', index) for index in range(2)]

    def sync():
        for device in devices:
            torch.cuda.synchronize(device)

    report = {'format': 'pose-opd-actor-variants-v1', 'host': socket.gethostname(),
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch_version': torch.__version__,
              'repo': str(args.repo), 'source_digest': source_digest(args.repo), 'args': vars(args).copy(),
              'sampler_backend': 'inverse_cdf', 'attention_precision': 'fp32', 'depth_cache_positions': 32,
              'variant_definitions': {
                  'baseline': 'Native fixed GEMV dispatch for all depth matrices.',
                  'linear_down': 'F.linear only for M=8 depth down-projections.',
                  'linear_all': 'F.linear for every M=8 decoder.lin1 call, including categorical heads.',
                  'tuned_down': 'Explicit winning native GEMV config for M=8 depth down-projections.',
                  'tuned_all': 'Explicit winning configs for M=8 trunk shapes; unmatched shapes retain baseline.'},
              'notes': ['Benchmark only: rows explicitly cropped to one minute; no training or checkpoint writes.',
                        'Only selected BF16 actor weights are resident; memory excludes learners, teachers, and Adam.',
                        '750-frame throughput is not a full-row estimate: temporal attention grows with context.',
                        'All candidates retain FP32 depth residuals/norm/RoPE/PxV and BF16 projection operands.',
                        'Different GEMM kernels can change BF16 rounding and therefore sampled trajectories.',
                        'Wall time includes prefill, GPU completion, and complete checked trajectory readback.'],
              'runs': []}

    def save():
        tmp = args.output.with_suffix(args.output.suffix + '.tmp')
        tmp.write_text(json.dumps(report, indent=2, default=str) + '\n')
        tmp.replace(args.output)

    def log(event, **values):
        print(json.dumps({'event': event, 'time': time.time(), **values}, default=str), flush=True)

    config = TrainingConfig.from_dict(json.loads(args.config.read_text()))
    spec = checkpoint_spec(config.student, 'student')
    report['student'] = spec
    report['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    log('data_begin')
    started = time.perf_counter()
    data = ProductionRows(config)
    items = data.items()
    # Same first actor pair as the40-rank effective4/GPU training plan.
    plan = make_plan(items, world=40, epochs=1, seed=config.seed, rows_per_rank=4)
    selected = plan['updates'][0][:8]
    rows = [benchmark_prefix(data.load(item, rank=index // 4), args.frames)
            for index, item in enumerate(selected)]
    seeds = [int(np.random.SeedSequence([config.seed, 0, index // 4, index % 4]).generate_state(1)[0])
             for index in range(8)]
    report['rows'] = [{'item': item, 'row_key': row['row_key'], 'prefix': int(row['P'][0]),
                       'frames': row['conversation_frames'], 'original_frames': row['benchmark_original_frames'],
                       'voice_clip': row['voice_clip'], 'seed': seed}
                      for item, row, seed in zip(selected, rows, seeds)]
    report['data_metadata'] = data.metadata
    report['data_seconds'] = time.perf_counter() - started
    log('data_ready', seconds=report['data_seconds'], rows=report['rows'])
    M = runtime.load_native()
    M.QWEN32, M.QWEN4B_CFG = args.qwen32_config, args.qwen4_config
    cfgs = (M.cfg32(spec['layers']), M.cfg4b(spec['tower_layers']), M.cfg06(spec['dec_layers']))
    with torch.device('meta'):
        shape_model = M.PPM(*cfgs, proj_r=spec['proj_r'], tower_vocab=spec['tower_vocab'],
                            pose_head=spec['pose_head'], decp_bb=True, vadapter=spec['vadapter'], nopose=False)
    log('model_build_begin')
    started = time.perf_counter()
    actor, layer_devices = build_actor(shape_model, cfgs, devices, spec['vadapter'],
                                        spec['decp_bb_alpha'], backbone_cut=args.cut)
    del shape_model
    log('model_load_begin')
    report['load_receipt'] = load_checkpoint(actor, Path(config.student), devices[0])
    sync()
    report['model_build_load_seconds'] = time.perf_counter() - started
    report['actor_parameter_bytes'] = {str(device): sum(p.numel() * p.element_size() for p in actor.parameters()
                                                        if p.device == device) for device in devices}
    report['exact_gain_theta'] = float(actor.vgain.theta.detach().cpu())
    log('model_ready', seconds=report['model_build_load_seconds'], gain=report['exact_gain_theta'])
    native = load_actor_native(M, args.repo)
    actor_type = make_fastppm_class(native)
    log('audio_features_begin')
    started = time.perf_counter()
    import audio_s4q
    front = audio_s4q.make_front({'s4_audio': 'flac', 's4_vox_model': args.voxtral,
                                's4_enc_hidden': 1280}, devices[0])
    features = []
    with torch.no_grad():
        for row in rows:
            latents = front.encode([torch.from_numpy(row['wav'])],
                                   [{'row_key': row['row_key'], 'n': row['conversation_frames']}])
            features.append(audio_features_from_latents(SimpleNamespace(m=actor, d0=devices[0]), latents).cpu())
            row.pop('wav')
            del latents
    del front
    gc.collect()
    report['post_front_cache_release'] = release_publication_cache(devices)
    report['audio_features_seconds'] = time.perf_counter() - started
    log('audio_features_ready', seconds=report['audio_features_seconds'])
    capacity = args.capacity_frames or max(row['conversation_frames'] for row in rows)
    if capacity < max(row['conversation_frames'] for row in rows):
        raise ValueError('Cache capacity must cover every truncated row')
    prefix = max(int(row['P'][0]) for row in rows)
    tuned = {}
    first_baseline = None

    def explicit_gemv(x, w, parameters):
        bn, bk, warps, stages = parameters
        shape, k, n = x.shape, x.shape[-1], w.shape[0]
        flat = x.reshape(-1, k).contiguous()
        y = torch.empty((len(flat), n), dtype=torch.bfloat16, device=x.device)
        with torch.cuda.device(x.device):
            native.LK._gemv_k.fn[(triton.cdiv(n, bn),)](flat, w, y, len(flat), n, k,
                BM=16, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
        return y.view(*shape[:-1], n)

    def tune(engine):
        # The raw LK wrapper launches on the current device; all tuning inputs
        # and events deliberately live on the first GPU.
        torch.cuda.set_device(devices[0])
        configs = ((16,128,4,3), (16,256,4,3), (32,128,4,3), (32,256,4,3),
                   (64,128,4,3), (16,512,4,2), (32,64,4,4), (16,128,2,4), (64,64,4,4))
        records = []
        decoder = engine.dec_p
        for key in ('wqkv', 'wo', 'wgu', 'wd'):
            weights = [layer[key] for layer in decoder.layers]
            n, k = weights[0].shape
            if (n, k) in tuned:
                continue
            x = torch.randn(8, k, dtype=torch.bfloat16, device=devices[0])
            native_outputs = [native.LK.gemv(x, weight) for weight in weights]
            choices = []
            for parameters in configs:
                outputs = [explicit_gemv(x, weight, parameters) for weight in weights]
                error = max(float((actual.float()-expected.float()).abs().max())
                            for actual, expected in zip(outputs, native_outputs))
                unequal = sum(int((actual != expected).sum()) for actual, expected in zip(outputs, native_outputs))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = [explicit_gemv(x, weight, parameters) for weight in weights]
                for _ in range(4):
                    graph.replay()
                torch.cuda.synchronize(devices[0])
                samples = []
                for _ in range(5):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(20):
                        graph.replay()
                    end.record(); end.synchronize()
                    samples.append(begin.elapsed_time(end)*1000/(20*len(weights)))
                choices.append({'parameters': parameters, 'microseconds': statistics.median(samples),
                                'max_abs_vs_native': error, 'unequal': unequal})
                del graph, outputs
            winner = min(choices, key=lambda choice: choice['microseconds'])
            tuned[(n,k)] = winner['parameters']
            records.append({'key': key, 'shape': [n,k], 'weight_count': len(weights),
                            'weight_bytes': sum(w.numel()*w.element_size() for w in weights),
                            'winner': winner, 'candidates': choices})
            log('gemv_tuned', **records[-1])
            del x, native_outputs
        report['gemv_tuning'] = records
        save()

    for run_index, variant in enumerate(variants):
        log('variant_build_begin', index=run_index, variant=variant)
        started = time.perf_counter()
        engine = actor_type(actor, layer_devices, cfgs, devices, n_inst=8, sampler_backend='inverse_cdf',
                            depth_cache_positions=32, attention_precision='fp32',
                            max_s=capacity/12.5, p_max=prefix+8)
        if args.tune_gemv and not tuned:
            tune(engine)
        if variant != 'baseline':
            for decoder in (engine.dec_s, engine.dec_p):
                default = decoder.lin1
                down_shape = tuple(decoder.layers[0]['wd'].shape)
                def linear(x, w, default=default, down_shape=down_shape, variant=variant):
                    rows_count = x.numel() // x.shape[-1]
                    selected = rows_count == 8 and (variant.endswith('all') or tuple(w.shape) == down_shape)
                    if selected and variant.startswith('linear'):
                        return F.linear(x, w)
                    if selected and variant.startswith('tuned') and tuple(w.shape) in tuned:
                        return explicit_gemv(x, w, tuned[tuple(w.shape)])
                    return default(x, w)
                decoder.lin1 = linear
        for index, (row, feature) in enumerate(zip(rows, features)):
            engine.set_audio_features(feature, prefix=int(row['P'][0]), instance=index)
        sync()
        build_seconds = time.perf_counter() - started
        log('variant_capture_begin', index=run_index, variant=variant)
        started = time.perf_counter()
        engine.capture(f=25, P=prefix)
        sync()
        capture_seconds = time.perf_counter() - started
        if args.profile_stages:
            engine.enable_stage_profile(32)
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        sync()
        log('variant_rollout_begin', index=run_index, variant=variant)
        started = time.perf_counter()
        outputs = rollout_rows(engine, rows, seeds=seeds, graphs=True,
            progress=lambda **fields: log('progress', index=run_index, variant=variant, **fields))
        sync()
        elapsed = time.perf_counter() - started
        hashes = {stream: hashlib.sha256(b''.join(np.ascontiguousarray(getattr(value, stream)).tobytes()
                   for value in outputs)).hexdigest() for stream in ('text','audio','pose')}
        if first_baseline is None and variant == 'baseline':
            first_baseline = {stream: [np.array(getattr(value, stream), copy=True) for value in outputs]
                              for stream in ('text', 'audio', 'pose')}
        differences = None
        if first_baseline is not None:
            differences = {}
            for stream in ('text', 'audio', 'pose'):
                unequal = total = 0
                for reference, value in zip(first_baseline[stream], outputs):
                    actual = getattr(value, stream)
                    mask = value.sampled_masks[stream]
                    unequal += int(np.count_nonzero(reference[mask] != actual[mask]))
                    total += int(actual[mask].size)
                differences[stream] = {'unequal_sampled_tokens': unequal, 'sampled_tokens': total,
                                       'unequal_fraction': unequal / max(total, 1)}
        memory = [{'device': str(device), 'allocated_gib': torch.cuda.memory_allocated(device)/2**30,
                   'reserved_gib': torch.cuda.memory_reserved(device)/2**30,
                   'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/2**30} for device in devices]
        steps = max(row['conversation_frames'] for row in rows) + 2 - config.prefill_frames
        record = {'index': run_index, 'variant': variant, 'build_seconds': build_seconds,
                  'capture_seconds': capture_seconds, 'rollout_seconds': elapsed,
                  'batched_steps': steps, 'ms_per_batched_step': elapsed*1000/steps,
                  'aggregate_row_frames_per_second': sum(row['conversation_frames']+2-config.prefill_frames
                                                        for row in rows)/elapsed,
                  'fingerprints': hashes, 'memory': memory,
                  'vs_first_baseline_trajectories': differences,
                  'stage_profile': engine.stage_profile_result() if args.profile_stages else None}
        if args.probe_depth:
            log('same_root_depth_probe_begin', index=run_index, variant=variant)
            record['same_root_depth_probe'] = same_root_depth_probe(engine, rows, runtime)
            log('same_root_depth_probe_complete', index=run_index, variant=variant,
                result=record['same_root_depth_probe'])
        report['runs'].append(record)
        save()
        log('variant_complete', **record)
        engine.close()
        engine.graphs.clear()
        del engine, outputs
        # A loop variable otherwise retains the final decoder and its packed weights.
        if 'decoder' in locals():
            del decoder
        if 'linear' in locals():
            del linear, default
        gc.collect()
        released = release_publication_cache(devices)
        log('variant_released', index=run_index, **released)
    report['complete'] = True
    save()


def same_root_depth_probe(engine, rows, runtime):
    """Compare decoder implementations with exactly equal hv/zb and32 codes.

    Called after synchronized timing/readback. This changes decoder KV scratch
    only and never model weights. Only rows reaching the batch's last frame
    contribute; short rows' artificial continuation slots are excluded.
    """
    import numpy as np
    import torch
    started = time.perf_counter()
    last_frame = max(row['conversation_frames'] for row in rows) + 1
    eligible = [index for index, row in enumerate(rows) if row['conversation_frames'] + 1 == last_frame]
    result = {'frame': last_frame, 'instances': eligible,
              'interpretation': 'Identical tower roots, D33 roots, and actual32 sampled codebooks; isolates depth arithmetic from temporal drift.',
              'streams': {}}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for stream, decoder, hv_source, zb_source, device, code_buffer in (
            ('audio', engine.dec_s, engine.hv_s, None, engine.dl, engine.depB),
            ('pose', engine.dec_p, engine.hv_p[:engine.NI], engine.zb_p[:engine.NI], engine.dp, engine.pdepB)):
            with torch.cuda.device(device):
                hv = hv_source.detach().clone()
                zb = None if zb_source is None else zb_source.detach().clone()
                positions = engine.Tv[device][:engine.NI]
                global_codes = code_buffer[engine.arI[device], positions].detach().clone()
                codes = global_codes - torch.arange(32, device=device) * 2048
                if not bool(((codes >= 0) & (codes < 2048)).all()):
                    raise ValueError('Final-frame probe codebooks are invalid')
                flat = torch.arange(engine.NI, device=device)
                reference_x, _ = runtime._decoder_inputs(decoder.dec, hv, global_codes, flat, zb)
                x = torch.cat([decoder.slot0(hv, zb)[:, None], decoder.dec.cb0_emb(codes[:,0])[:,None].float()], 1)
                if not torch.equal(x, reference_x[:,:2]):
                    raise RuntimeError('Probe roots differ before the depth transformer')
                fast_logits = []
                for codebook in range(1,32):
                    fast_logits.append(decoder.head(decoder.step(x, codebook-1), codebook).float())
                    if codebook < 31:
                        x = decoder.emb(codes[:,codebook], codebook)
                fast_logits = torch.stack(fast_logits, 1)
                cos, sin = decoder.dec.model.rotary_emb(reference_x,
                    torch.arange(reference_x.shape[1], device=device)[None])
                hidden = reference_x
                for layer in decoder.dec.model.layers:
                    hidden = runtime._depth_layer(layer, hidden, cos, sin)
                hidden = runtime._depth_norm(decoder.dec.model.norm, hidden)
                reference_logits = torch.einsum('ndh,dvh->ndv', hidden[:,1:],
                    decoder.dec.heads.weight.view(31,2048,-1)).float()
                selected = torch.tensor(eligible, device=device)
                fast = fast_logits.index_select(0, selected).double()
                reference = reference_logits.index_select(0, selected).double()
                logp, logq = fast.log_softmax(-1), reference.log_softmax(-1)
                kl = (logp.exp()*(logp-logq)).sum(-1)
                reverse = (logq.exp()*(logq-logp)).sum(-1)
                delta = (fast-reference).abs()
                def metrics(index=None):
                    a = kl if index is None else kl[:,index]
                    b = reverse if index is None else reverse[:,index]
                    d = delta if index is None else delta[:,index]
                    return {'actor_to_replay_kl_mean': float(a.mean()), 'actor_to_replay_kl_max': float(a.max()),
                            'replay_to_actor_kl_mean': float(b.mean()), 'logit_mean_abs': float(d.mean()),
                            'logit_max_abs': float(d.max())}
                result['streams'][stream] = {'all31': metrics(),
                    'by_codebook': {str(book): metrics(book-1) for book in (1,14,31)},
                    'tower_root_rms': float(hv.float().square().mean().sqrt()),
                    'd33_root_rms': None if zb is None else float(zb.float().square().mean().sqrt()),
                    'slot0_rms': float(reference_x[:,0].square().mean().sqrt()),
                    'root_equal': True, 'input_dtype': str(reference_x.dtype)}
    for device in engine.devs:
        torch.cuda.synchronize(device)
    result['seconds'] = time.perf_counter() - started
    return result


if __name__ == '__main__':
    main()
