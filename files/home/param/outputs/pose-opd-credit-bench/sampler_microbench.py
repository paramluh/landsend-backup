import datetime
import json
import os
from pathlib import Path
import statistics
import time

import torch

wall_start = time.time()
torch.set_num_threads(1)
torch.manual_seed(221)
device = torch.device('cuda:0')
torch.cuda.set_device(device)
generators = [torch.Generator(device=device).manual_seed(1000 + row) for row in range(8)]
rows = []
gpu_milliseconds = 0.
torch.cuda.synchronize()
gpu_start = time.time()
for vocabulary in (2048, 151682):
    logits = torch.randn(8, vocabulary, device=device, dtype=torch.float32) * 5
    # Exactly the production policy weights; only the CDF accumulation changes.
    weights = logits.log_softmax(-1).exp()
    uniforms64 = torch.rand(32, 8, 1, device=device, dtype=torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
    uniforms32 = uniforms64.float().clamp_(min=torch.finfo(torch.float32).tiny,
                                        max=1 - torch.finfo(torch.float32).eps / 2)
    cdf64 = weights.cumsum(-1, dtype=torch.float64)
    cdf32 = weights.cumsum(-1, dtype=torch.float32)
    quality = {}
    for name, cumulative in (('fp32', cdf32), ('fp64', cdf64)):
        delta = torch.diff(cumulative, prepend=torch.zeros_like(cumulative[:, :1]), dim=-1)
        unreachable = (delta == 0) & (weights > 0)
        quality[name] = dict(positive_weights_collapsed_fraction=float(unreachable.float().mean()),
            collapsed_probability_mass_per_row=[float(v) for v in (weights.double() * unreachable).sum(-1)],
            max_cdf_absolute_error_vs_fp64=float((cumulative.double() - cdf64).abs().max()))

    def multinomial(index):
        return torch.cat([torch.multinomial(weights[row:row+1], 1, generator=generator)[:, 0]
                          for row, generator in enumerate(generators)], 0)

    def cdf_sampler(index, accumulation, uniform_dtype):
        cumulative = weights.cumsum(-1, dtype=accumulation)
        uniform = uniforms64[index] if uniform_dtype == torch.float64 else uniforms32[index]
        thresholds = uniform * cumulative[:, -1:]
        return torch.searchsorted(cumulative, thresholds, right=False).squeeze(-1)

    variants = [('current_row_multinomial', multinomial)]
    for accumulation in (torch.float32, torch.float64):
        for uniform_dtype in (torch.float32, torch.float64):
            label = f'cdf_{str(accumulation).split(".")[-1]}_u_{str(uniform_dtype).split(".")[-1]}'
            variants.append((label, lambda index, a=accumulation, u=uniform_dtype: cdf_sampler(index, a, u)))
    for name, sampler in variants:
        for index in range(2):
            sampler(index)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        if name == 'current_row_multinomial':
            for generator in generators:
                graph.register_generator_state(generator)
        with torch.cuda.graph(graph):
            sampled = [sampler(index) for index in range(32)]
        observations = []
        for _ in range(3):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            elapsed = begin.elapsed_time(end)
            gpu_milliseconds += elapsed
            observations.append(elapsed * 1000 / 32)
        rows.append(dict(vocabulary=vocabulary, batch=8, variant=name, graph_repetitions=32,
                         latency_us=dict(median=statistics.median(observations), min=min(observations), max=max(observations)),
                         sampled_in_range=all(bool(((s >= 0) & (s < vocabulary)).all()) for s in sampled),
                         probability_quality=quality if name == 'current_row_multinomial' else None))
        del graph, sampled
    del weights, logits, uniforms32, uniforms64, cdf32, cdf64
torch.cuda.synchronize()
gpu_end = time.time()
peak = torch.cuda.max_memory_allocated()
report = dict(host=os.uname().nodename, physical_gpu=7, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    wall_start_unix=wall_start, gpu_interval_start_unix=gpu_start, gpu_interval_end_unix=gpu_end,
    wall_end_unix=time.time(), measured_graph_gpu_milliseconds=gpu_milliseconds,
    max_pytorch_allocated_bytes=peak, max_pytorch_reserved_bytes=torch.cuda.max_memory_reserved(),
    device=torch.cuda.get_device_name(), torch_version=torch.__version__, rows=rows,
    notes=['Bounded microprofile colocated with v4 actor rank38; exclude this wall interval from that actor throughput.',
           'Eight independent perrow torch.multinomial generators versus one batched cumsum/searchsorted with precomputed uniforms.',
           'All variants use identical FP32 log_softmax(logits).exp weights; CDF thresholds are uniform times final CDF total.',
           'FP32 and FP64 uniforms use the same underlying random uniforms rounded to their dtype; no RNG cost in CDF timing.',
           'Sharp random logits use standard deviation5. Probability-collapse test detects positive weights receiving zero CDF interval.'])
destination = Path('/home/param/outputs/pose-opd-credit-bench/sampler-m8.json')
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
