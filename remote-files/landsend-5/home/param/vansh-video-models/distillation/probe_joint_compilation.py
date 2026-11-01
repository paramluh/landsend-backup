"""Bounded CUDA specialization probe; the five-node trainer measures real throughput.

Run: python -m distillation.probe_joint_compilation --backend eager --output /tmp/joint-layouts.json
Use --backend inductor to additionally compile/run forward and backward kernels.
The tiny fixture includes blocks with and without audio, frozen teacher scoring,
critic backward, detached short-prefix pools, and six-prefix student backward.
"""

import argparse
import copy
import json
from pathlib import Path
import time

import torch
from torch._dynamo.utils import counters

import distillation.bimodal_net as bimodal_net
from distillation.joint_dmd import generate_resolver, generate_stabilizer, score_resolver, score_stabilizer
from distillation.test_joint_layout import rows_and_noise, small_net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('eager', 'inductor'), default='inductor')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error('At least two iterations are needed to check stable specialization counts')
    torch.set_num_threads(4)
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 256
    torch.manual_seed(79)
    net = small_net('cuda')
    net.audio_blocks[1] = None  # Exercise the real alternating block variants.
    teacher = copy.deepcopy(net).requires_grad_(False)
    bimodal_net._WHOLE['default'] = torch.compile(bimodal_net.block_forward, backend=args.backend, dynamic=False)
    for model in (net, teacher):
        model.configure(True, 'block', compile_mode='block')
    rows, noise = rows_and_noise('cuda', batch=1)
    rows = {key: value.detach() for key, value in rows.items()}
    keep = torch.tensor([[True, False, True, False, True, True]], device='cuda')
    times = torch.tensor([[.3, .5, .7, .9]], device='cuda')
    results = []

    def phase(name, iteration):
        print(json.dumps(dict(iteration=iteration, phase=name, stats=dict(counters['stats']))), flush=True)

    for iteration in range(args.iterations):
        start = time.monotonic()
        net.zero_grad(set_to_none=True)
        with torch.no_grad():
            pool, context = generate_stabilizer(net, rows, noise=noise[:, :, -7:].detach(),
                                                history_keep=keep, prefix_frames=3)
            phase('pool', iteration)
            resolved = generate_resolver(net, rows, noise=noise[:, :, -4:].detach())
            score_stabilizer(teacher, rows, pool, times, context, 2.5)
            score_resolver(teacher, rows, resolved, times*.2)
        phase('frozen_scores', iteration)
        critic = score_stabilizer(net, rows, pool, times, context)
        resolver = score_resolver(net, rows, resolved, times*.2)
        (critic.square().mean()+resolver.square().mean()).backward()
        phase('critic_backward', iteration)
        net.zero_grad(set_to_none=True)
        output, _ = generate_stabilizer(net, rows, noise=noise.detach(), history_keep=keep)
        resolved = generate_resolver(net, rows, noise=noise[:, :, -4:].detach())
        (output.square().mean()+resolved.square().mean()).backward()
        torch.cuda.synchronize()
        result = dict(iteration=iteration, seconds=time.monotonic()-start,
                      counters={key: dict(value) for key, value in counters.items()},
                      peak_allocated=torch.cuda.max_memory_allocated())
        results.append(result)
        print(json.dumps(result), flush=True)
        keep.logical_not_()
        rows['ref_keep'].logical_not_()
    report = dict(backend=args.backend, fixture='two-layer tiny network, audio and no-audio blocks',
                  results=results, stable=results[-1]['counters']['stats']['unique_graphs'] ==
                  results[0]['counters']['stats']['unique_graphs'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if not report['stable']:
        raise RuntimeError('New graphs appeared after the first complete joint cycle')


if __name__ == '__main__':
    main()
