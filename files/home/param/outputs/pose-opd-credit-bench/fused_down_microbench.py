"""Isolated experimental kernel; never imported by the training package."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_kernel(GU, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    mi = tl.arange(0, 16)
    acc = tl.zeros((16, BN), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        g = tl.load(GU + mi[:, None] * (2 * K) + k[None, :],
                    mask=mi[:, None] < M, other=0.).to(tl.float32)
        u = tl.load(GU + mi[:, None] * (2 * K) + K + k[None, :],
                    mask=mi[:, None] < M, other=0.).to(tl.float32)
        activation = (g / (1. + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        x = (activation * u).to(tl.bfloat16)
        w = tl.load(W + n[:, None].to(tl.int64) * K + k[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(Y + mi[:, None] * N + n[None, :], acc.to(tl.bfloat16),
             mask=mi[:, None] < M)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    sys.path.insert(0, str(Path(args.repo) / 'infer/engine/ppm'))
    import light_kernels as LK
    torch.manual_seed(20261031)
    m, n, k, count = 8, 1024, 3072, 32
    xs = [torch.randn(m, 2 * k, device='cuda', dtype=torch.bfloat16) for _ in range(count)]
    ws = [torch.randn(n, k, device='cuda', dtype=torch.bfloat16) / math.sqrt(k) for _ in range(count)]
    reference = [LK.gemv(LK.silu_mul_gu(x), w) for x, w in zip(xs, ws)]

    def fused(x, w, bn, bk, warps, stages):
        y = torch.empty(m, n, device=x.device, dtype=x.dtype)
        fused_kernel[(n // bn,)](x, w, y, m, n, k, bn, bk,
                                 num_warps=warps, num_stages=stages)
        return y

    candidates = [('native_silu_plus_gemv', lambda x, w: LK.gemv(LK.silu_mul_gu(x), w)),
                  ('native_silu_plus_linear', lambda x, w: F.linear(LK.silu_mul_gu(x), w))]
    for bn, bk, warps, stages in ((16, 128, 4, 3), (16, 256, 4, 3), (16, 512, 4, 2),
                                  (32, 128, 4, 3), (32, 256, 4, 3), (64, 128, 4, 3),
                                  (16, 128, 2, 4), (32, 64, 4, 4)):
        candidates.append((f'fused_bn{bn}_bk{bk}_w{warps}_s{stages}',
                           lambda x, w, bn=bn, bk=bk, warps=warps, stages=stages:
                           fused(x, w, bn, bk, warps, stages)))
    results = []
    for name, fn in candidates:
        outputs = [fn(x, w) for x, w in zip(xs, ws)]
        differences = torch.cat([(a.float() - b.float()).flatten() for a, b in zip(outputs, reference)])
        parity = {'max_abs_difference': float(differences.abs().max()),
                  'mean_abs_difference': float(differences.abs().mean()),
                  'different_fraction': float((differences != 0).float().mean())}
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = [fn(x, w) for x, w in zip(xs, ws)]
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                graph.replay()
            stop.record(); stop.synchronize()
            samples.append(start.elapsed_time(stop) * 1000 / (20 * count))
        result = {'variant': name, 'microseconds_per_layer': samples, **parity}
        results.append(result)
        print(json.dumps(result), flush=True)
        del graph, graph_outputs, outputs
    Path(args.output).write_text(json.dumps({'gpu': torch.cuda.get_device_name(), 'shape': [m, n, k],
        'distinct_weights': count, 'weight_bytes': count * n * k * 2,
        'time': time.time(), 'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
