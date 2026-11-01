import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

wall_start = time.time()
source = Path('/home/param/pose-worktrees/opd-bench-actor8-ladder-v4/infer/engine/ppm/light_exact_ppm.py')
sys.path.insert(0, str(source.parent))
import torch
import light_exact_ppm as LX

torch.set_num_threads(1)
torch.cuda.set_device(0)
torch.manual_seed(317)
batch, nh, kvh, dim = 8, 16, 8, 128
width = (nh + 2 * kvh) * dim
rows, parity = [], []
gpu_milliseconds = 0.
gpu_start = time.time()
for decoder_step in (0, 1, 15, 30):
    length = 2 if decoder_step == 0 else 1
    position = 0 if decoder_step == 0 else decoder_step + 1
    for seed in range(4):
        torch.manual_seed(1000 + seed + decoder_step * 10)
        qkv = torch.randn(batch * length, width, device='cuda', dtype=torch.bfloat16)
        qnorm = (1 + .1 * torch.randn(dim, device='cuda')).bfloat16().float()
        knorm = (1 + .1 * torch.randn(dim, device='cuda')).bfloat16().float()
        theta = torch.arange(position, position + length, device='cuda')[:, None] * (
            10000 ** (-torch.arange(0, dim, 2, device='cuda').float() / dim))[None]
        phases = torch.cat((theta, theta), dim=-1)
        cos, sin = phases.cos().contiguous(), phases.sin().contiguous()
        key = torch.randn(batch, 64, kvh, dim, device='cuda', dtype=torch.bfloat16)
        value = torch.randn_like(key)
        caches = {64: (key.clone(), value.clone()),
                  32: (key[:, :32].contiguous(), value[:, :32].contiguous())}

        def run(capacity):
            keys, values = caches[capacity]
            return LX.attn_exact(qkv, qnorm, knorm, 1e-6, cos, sin, keys, values,
                                 position, length, nh, kvh, dim, dim ** -.5, round_p=1)

        outputs = {capacity: run(capacity) for capacity in (64, 32)}
        diff = outputs[64].float() - outputs[32].float()
        parity.append(dict(decoder_step=decoder_step, seed=seed, length=length, position=position,
            output_equal=bool(torch.equal(outputs[64], outputs[32])), max_abs=float(diff.abs().max()),
            key_cache_equal=bool(torch.equal(caches[64][0][:, :32], caches[32][0])),
            value_cache_equal=bool(torch.equal(caches[64][1][:, :32], caches[32][1]))))
        if seed == 0:
            for capacity in (64, 32):
                run(capacity)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    results = [run(capacity) for _ in range(32)]
                elapsed = []
                for _ in range(3):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    measured = start.elapsed_time(end)
                    gpu_milliseconds += measured
                    elapsed.append(measured * 1000 / 32)
                rows.append(dict(batch=batch, decoder_step=decoder_step, length=length, position=position,
                    cache_positions=capacity, latency_us=dict(median=statistics.median(elapsed),
                        min=min(elapsed), max=max(elapsed))))
                del graph, results
        del outputs, caches, key, value, qkv, cos, sin, qnorm, knorm
torch.cuda.synchronize()
gpu_end = time.time()
report = dict(host=os.uname().nodename, pid=os.getpid(), physical_gpu=7,
    source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    wall_start_unix=wall_start, gpu_interval_start_unix=gpu_start,
    gpu_interval_end_unix=gpu_end, wall_end_unix=time.time(),
    max_pytorch_allocated_bytes=torch.cuda.max_memory_allocated(),
    max_pytorch_reserved_bytes=torch.cuda.max_memory_reserved(),
    measured_graph_gpu_milliseconds=gpu_milliseconds, rows=rows, parity=parity,
    bitwise_parity_all=all(p['output_equal'] and p['key_cache_equal'] and p['value_cache_equal'] for p in parity),
    notes=['Bounded colocated microprofile on landsend12 GPU7; exclude rank38 actor interval.',
           'Actual LX.attn_exact kernel, FP32 norm/RoPE and BF16 QKV/KV/output; ROUND_P=1.',
           'Identical QKV and valid KV history across64- and32-position caches,4randomseeds perstep.',
           'Each measured CUDAgraph repeats32kernelcalls; absolute timing is affected by actor co-location.'])
destination = Path('/home/param/outputs/pose-opd-credit-bench/attention-cache32-vs64.json')
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
