"""Diagnostic: exact natural depth decoder, full vs two GPU row halves."""
import inspect
import json
import pathlib
import statistics
import sys
import time
from types import MethodType

import torch

sys.path.insert(0, str(pathlib.Path.cwd()))
from finetune.pose_opd_credit import runtime
from finetune.pose_opd_credit.rollout import make_fastppm_class, _TUNED_M16
from finetune.pose_opd_credit.train import load_actor_native

torch.set_grad_enabled(False)
torch.manual_seed(812763)
ROOT = pathlib.Path(__file__).parent
CHECKPOINT = pathlib.Path('/home/param/models/imported-weights/ppm-aloe-v2vox-raw-w14503/export-253')
M = runtime.load_native()
native = load_actor_native(M)
closure = inspect.getclosurevars(make_fastppm_class(native).__init__).nonlocals
gen = closure['natural_depth_gen']
step = closure['fp32_attention_depth_step']


def log(event, **kw):
    print(json.dumps({'event': event, 'time': time.time(), **kw}), flush=True)


def load_decoder(device):
    cfg = M.cfg06()
    with torch.device('meta'):
        decoder = M.Decoder(2560, cfg).to(dtype=torch.bfloat16)
    decoder.to_empty(device=device)
    decoder.model.rotary_emb = type(decoder.model.rotary_emb)(config=cfg, device=device)
    offsets = {name: 0 for name, _ in decoder.named_parameters()}
    params = dict(decoder.named_parameters())
    world = json.loads((CHECKPOINT / 'meta.json').read_text())['world']
    for rank in range(world):
        state = torch.load(CHECKPOINT / f'rank{rank}.pt', mmap=True, weights_only=True, map_location='cpu')
        for key, value in state.items():
            if not key.startswith('dec_s.'):
                continue
            name = key[len('dec_s.'):]
            n = value.shape[0]
            params[name][offsets[name]:offsets[name] + n].copy_(value)
            offsets[name] += n
        del state
    state = torch.load(CHECKPOINT / 'rep.pt', mmap=True, weights_only=True, map_location='cpu')
    for key, value in state.items():
        if key.startswith('dec_s.'):
            name = key[len('dec_s.'):]
            params[name].copy_(value)
            offsets[name] = value.shape[0]
    assert all(offsets[name] == p.shape[0] for name, p in params.items())
    return decoder.eval().requires_grad_(False)


def masked_tuned(fallback, x, w):
    rows = x.numel() // x.shape[-1]
    config = _TUNED_M16.get(tuple(w.shape)) if rows <= 16 else None
    if config is None or x.dtype != torch.bfloat16 or not w.is_contiguous():
        return fallback(x, w)
    bn, bk, warps, stages = config
    n, k = w.shape
    out = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    with torch.cuda.device(x.device):
        native.LK._gemv_k.fn[((n + bn - 1) // bn,)](
            x.reshape(rows, k).contiguous(), w, out, rows, n, k,
            BM=16, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    return out


def actor(decoder, count, device, hv, codes, uniforms, tuned=False):
    with torch.cuda.device(device):
        fused = native.FusedDec(decoder, device, count, max_pos=32)
        fused.gen = MethodType(gen, fused)
        fused.step = MethodType(step, fused)
        if tuned:
            fallback = fused.lin1
            fused.lin1 = lambda x, w: masked_tuned(fallback, x, w)
        fused._opd_uniforms = uniforms.to(device).clone()
        fused._opd_positions = torch.zeros(count, dtype=torch.int64, device=device)
        fused._opd_rows = torch.arange(count, device=device)
        fused._opd_logprobs = torch.zeros(count, 32, device=device)
        fused._opd_probe_logits = None
        h = hv.to(device).clone()
        c = codes.to(device).clone()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                output = fused.gen(h, c, None, None, None)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = fused.gen(h, c, None, None, None)
        return {'fused': fused, 'hv': h, 'cb0': c, 'output': output, 'graph': graph,
                'stream': stream, 'device': device}


def sync():
    for device in (0, 1):
        torch.cuda.synchronize(device)


def run_single(a):
    with torch.cuda.stream(a['stream']):
        a['graph'].replay()


def run_split(halves, source_h, source_c, output, logps):
    # Include actual peer-copy inputs and gathering. No CPU fence between GPUs.
    n = source_h.shape[0] // 2
    events = []
    for i, a in enumerate(halves):
        with torch.cuda.stream(a['stream']):
            a['hv'].copy_(source_h[i * n:(i + 1) * n], non_blocking=True)
            a['cb0'].copy_(source_c[i * n:(i + 1) * n], non_blocking=True)
            a['graph'].replay()
            event = torch.cuda.Event()
            event.record()
            events.append(event)
    with torch.cuda.stream(halves[1]['stream']):
        for i, (a, event) in enumerate(zip(halves, events)):
            halves[1]['stream'].wait_event(event)
            output[i * n:(i + 1) * n].copy_(a['output'], non_blocking=True)
            logps[i * n:(i + 1) * n].copy_(a['fused']._opd_logprobs, non_blocking=True)


def timing(fn):
    for _ in range(3):
        fn()
    sync()
    trials = []
    for _ in range(12):
        start = time.perf_counter()
        fn()
        sync()
        trials.append((time.perf_counter() - start) * 1000)
    return {'median_ms': statistics.median(trials), 'min_ms': min(trials), 'trials_ms': trials}


def main():
    log('load_begin')
    dec1 = load_decoder(torch.device('cuda:1'))
    dec0 = load_decoder(torch.device('cuda:0'))
    log('weights_loaded', bytes=sum(p.numel() * p.element_size() for p in dec1.parameters()))
    records = []
    for count in ((24,) if '--probe' in sys.argv else (24, 48, 16)):
        hv = torch.randn(count, 2560, device='cuda:1', dtype=torch.bfloat16)
        c0 = torch.randint(2048, (count,), device='cuda:1')
        uniforms = torch.rand(count, 1, 32, dtype=torch.float64)
        sync()
        full = actor(dec1, count, torch.device('cuda:1'), hv, c0, uniforms,
                     tuned=count == 16)
        baseline = timing(lambda: run_single(full))
        log('baseline', batch=count, **baseline)
        for tuned in (False, True):
            half = count // 2
            halves = [actor(dec, half, torch.device(f'cuda:{i}'),
                            hv[i * half:(i + 1) * half], c0[i * half:(i + 1) * half],
                            uniforms[i * half:(i + 1) * half], tuned=tuned)
                      for i, dec in enumerate((dec0, dec1))]
            output = torch.empty_like(full['output'])
            logps = torch.empty_like(full['fused']._opd_logprobs)
            measured = timing(lambda: run_split(halves, hv, c0, output, logps))
            # Same fixed uniforms, row order and causal codebook sequence.
            mismatches = int((full['output'] != output).sum())
            lp_max = float((full['fused']._opd_logprobs - logps).abs().max())
            record = {'batch': count, 'split': [half, half], 'masked_tuned': tuned,
                      'baseline': baseline, 'split_timing': measured,
                      'speedup': baseline['median_ms'] / measured['median_ms'],
                      'sampled_token_mismatches': mismatches, 'behavior_max_abs': lp_max}
            if '--probe' in sys.argv:
                # Compare identical causal inputs, rather than diverged samples.
                truth = full['output'].clone()
                with torch.cuda.device(full['device']):
                    ref = full['fused'].teacher_logits(full['hv'], truth)
                fixed = []
                for i, a in enumerate(halves):
                    with torch.cuda.device(a['device']):
                        fixed.append(a['fused'].teacher_logits(a['hv'],
                            truth[i * half:(i + 1) * half].to(a['device'])).to('cuda:1'))
                test = torch.cat(fixed)
                assert torch.isfinite(ref).all() and torch.isfinite(test).all()
                lp, lq = ref.log_softmax(-1), test.log_softmax(-1)
                kl = (lp.exp() * (lp - lq)).sum(-1)
                delta = ref - test
                record['forced_context_probe'] = {
                    'kl_max': float(kl.max()), 'kl_mean': float(kl.mean()),
                    'relative_rms': float(delta.square().mean().sqrt() / ref.square().mean().sqrt()),
                    'max_logit_abs': float(delta.abs().max()),
                    'argmax_match_fraction': float((ref.argmax(-1) == test.argmax(-1)).float().mean()),
                    'first_cb_kl_max': float(kl[:, 0].max()),
                    'first_cb_sample_mismatches': int((truth[:, 1] != output[:, 1]).sum()),
                }
                del truth, ref, fixed, test, lp, lq, kl, delta
            log('comparison', **record)
            records.append(record)
            (ROOT / 'results.json').write_text(json.dumps(records, indent=2))
            del halves, output, logps
            sync()
        del full
        sync()
    log('complete')


if __name__ == '__main__':
    main()
