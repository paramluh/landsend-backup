"""Bounded single-GPU FA3 parity check of absolute-position paged KV reuse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .streaming.paged_kv import allocate_cache


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--positions', type=int, default=2304)
    args = parser.parse_args(argv)
    if args.positions < 2048 or args.positions % 32:
        parser.error('Use at least2048 encoder positions, divisible by32')
    from . import runtime
    native = runtime.load_native()
    fa = native.fa3_op._mod().flash_attn_interface
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(20261031)
    shape = (args.batch, args.positions, 32, 64)
    q, k, v = [torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(3)]
    fk, fv, _, _ = allocate_cache(batch=args.batch, positions=args.positions, heads=32, head_dim=64,
                                  layers=1, device=device)
    pk, pv, table, layout = allocate_cache(batch=args.batch, positions=args.positions, heads=32,
                                         head_dim=64, layers=1, device=device, mode='paged')
    differences = []
    call_count = 0
    started = time.monotonic()

    def compare(start, count, *, mutated=False):
        nonlocal call_count
        sl = torch.full((args.batch,), start, device=device, dtype=torch.int32)
        args_kv = dict(k=k[:, start:start + count], v=v[:, start:start + count],
                       cache_seqlens=sl, causal=True, window_size=(749, 0))
        expected = fa.flash_attn_with_kvcache(q[:, start:start + count], fk[0], fv[0], **args_kv)
        actual = fa.flash_attn_with_kvcache(q[:, start:start + count], pk[0], pv[0], page_table=table, **args_kv)
        delta = (actual.float() - expected.float()).abs()
        differences.append(torch.stack([delta.max(), delta.square().mean()]))
        call_count += 1

    for start in range(0, args.positions, 4):
        compare(start, 4)
    # The high-water cache retains future speculative writes. Rewind eight
    # model frames, replace those keys/values, then append24+8 in causal blocks.
    rewind = args.positions - 32
    k[:, rewind:] *= -.5
    v[:, rewind:] += .25
    compare(rewind, 24, mutated=True)
    compare(rewind + 24, 8, mutated=True)
    stats = torch.stack(differences)
    torch.cuda.synchronize(device)
    result = {'status': 'passed', 'batch': args.batch, 'encoder_positions': args.positions,
              'full_logical_positions': args.positions, 'physical_capacity': layout.capacity_positions,
              'pages_per_row': layout.pages_per_row, 'page_size': layout.page_size,
              'attention_window': 750, 'rewind_positions': 32, 'block_append_tested': [24, 8],
              'comparisons': call_count, 'maximum_absolute_error': float(stats[:, 0].max()),
              'rms_error_maximum_call': float(stats[:, 1].max().sqrt()),
              'bitwise_equal_all_calls': bool((stats[:, 0] == 0).all()),
              'seconds': time.monotonic() - started,
              'paged_bytes_one_layer': sum(x.numel() * x.element_size() for x in pk + pv),
              'contiguous_bytes_one_layer': sum(x.numel() * x.element_size() for x in fk + fv)}
    if result['maximum_absolute_error'] > .03125 or result['rms_error_maximum_call'] > .002:
        result['status'] = 'failed'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise AssertionError(result)


if __name__ == '__main__':
    main()
