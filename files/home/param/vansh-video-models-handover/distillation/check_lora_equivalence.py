import argparse
import copy
import json
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from distillation.bimodal_dmd import UniformStabilizerRollout, stab_pass
from distillation.bimodal_net import BimodalNet
from distillation.lora import LoRALinear, inject_lora
from models.pose.audio import AudioAdapter
from models.wan.model import WanModel


TARGETS = [r'transformer\.blocks\.\d+\.(?:self_attn\.(?:q|k|v|o)|ffn\.(?:0|2))',
           r'audio_adapter\..*(?:q_proj|kv_proj|out_proj)']


def build_small_net(device):
    transformer = WanModel(patch_size=(1, 1, 1), in_dim=2, out_dim=2, dim=128, ffn_dim=256,
                           freq_dim=64, num_heads=2, num_layers=2)
    if device.type == 'cuda':
        transformer.bfloat16()
    encoder = nn.Module()
    encoder.null_token = nn.Parameter(torch.randn(1, 1, 1, 128))
    encoder.no_audio_tokens = nn.Parameter(torch.randn(1, 1, 3, 128))
    adapter = AudioAdapter(hidden_dim=128, num_attn_heads=2, num_layers=2,
                           injection_offset=1, injection_interval=1, initialization='random')
    return BimodalNet(transformer, encoder, adapter, nn.Conv3d(2, 128, 1), nn.Parameter(torch.randn(128)), (2, 2), 2).to(device)


def attention_cpu(q, k, v):
    output = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float())
    return output.transpose(1, 2).to(q.dtype)


def run(net, source, workload, gradients, zero_adapters):
    values = {name: value.detach().clone().requires_grad_(gradients) for name, value in source.items()}
    latents, noise, tokens = values['latents'], values['noise'], values['tokens']
    rows = dict(latents=latents, pose=values['pose'], ofi=torch.arange(8, device=latents.device)[None].expand(2, -1),
                drop=torch.tensor([False, True], device=latents.device), history_t=.2)
    with torch.set_grad_enabled(gradients):
        if workload == 'rollout':
            output, _ = UniformStabilizerRollout([2] * 4, 2, .2, .4)(net, rows, tokens, noise, noise, stab_only=True)
        else:
            t = torch.tensor([.3, .5, .7, .9], device=latents.device)[None].expand(2, -1)
            x = .6 * latents[:, :, 4:] + .4 * noise[:, :, 4:]
            history = .8 * latents[:, :, 1:4] + .2 * noise[:, :, 1:4]
            audio = net.frame_audio(tokens, rows['ofi'][:, 4:], rows['drop'])
            mask = torch.ones(2, 4, dtype=torch.bool, device=latents.device)
            output = stab_pass(net, rows, x, t, history, audio, mask)
        if gradients:
            output.float().square().mean().backward()
    result = {'output': output.detach().float().cpu()}
    if gradients:
        result.update({f'gradient/{name}': value.grad.detach().float().cpu() for name, value in values.items()})
        adapters = [layer for layer in net.modules() if isinstance(layer, LoRALinear)]
        for layer in adapters:
            assert layer.lora_A.grad is not None
            assert layer.lora_B.grad is not None
            if zero_adapters:
                assert torch.count_nonzero(layer.lora_A.grad) == 0
        if adapters:
            assert any(torch.count_nonzero(layer.lora_B.grad) > 0 for layer in adapters)
            if not zero_adapters:
                assert any(torch.count_nonzero(layer.lora_A.grad) > 0 for layer in adapters)
            for factor in ('lora_A', 'lora_B'):
                result[f'gradient/{factor}'] = torch.cat(
                    [getattr(layer, factor).grad.detach().float().flatten().cpu() for layer in adapters])
    net.zero_grad(set_to_none=True)
    return result


def differences(actual, reference, rtol, atol):
    result = {}
    for name, value in actual.items():
        if name not in reference:
            continue
        expected = reference[name]
        delta = value - expected
        result[name] = dict(max_abs=delta.abs().max().item(), rms=delta.square().mean().sqrt().item(),
                            relative_rms=(delta.square().mean() / expected.square().mean().clamp_min(1e-30)).sqrt().item(),
                            aggregate_within_tolerance=(delta.square().mean().sqrt() <= atol + rtol * expected.square().mean().sqrt()).item(),
                            within_tolerance=torch.allclose(value, expected, rtol=rtol, atol=atol))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--cudagraphs', action='store_true')
    parser.add_argument('--fused-qkv', action='store_true')
    parser.add_argument('--production-only', action='store_true', help='compare only nonzero separate/fused adapters')
    parser.add_argument('--cache-dir', help='optional isolated root for persistent Inductor and Triton caches')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--rollout', action='store_true')
    parser.add_argument('--checkpoint', choices=('none', 'block'), default='none')
    parser.add_argument('--base-dtype', choices=('bfloat16',))
    parser.add_argument('--rtol', type=float, default=.02)
    parser.add_argument('--atol', type=float, default=.002)
    args = parser.parse_args()
    if args.production_only and not args.fused_qkv:
        parser.error('--production-only requires --fused-qkv')
    if args.cudagraphs and (not args.compile or args.device != 'cuda' or args.iterations < 3):
        parser.error('--cudagraphs requires --compile --device cuda and at least 3 iterations for replay')
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    if args.cache_dir:
        cache = Path(args.cache_dir).resolve()
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache / 'inductor')
        os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(117)
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    template = build_small_net(device)
    source = {name: torch.randn(2, 2, 8, 2, 2, device=device) for name in ('latents', 'pose', 'noise')}
    source['tokens'] = torch.randn(2, 16, 128, device=device)
    results, passed = {}, True
    with ExitStack() as stack:
        if device.type == 'cpu':
            stack.enter_context(patch('distillation.bimodal_net.flash_attn_func', attention_cpu))
            stack.enter_context(patch('models.pose.audio.flash_attn_func', attention_cpu))
            if args.base_dtype:
                stack.enter_context(torch.autocast('cpu', dtype=torch.bfloat16))
        modes = ['dense', 'frozen', 'lora'] + (['lora_nonzero', 'lora_fused_nonzero'] if args.fused_qkv else [])
        if args.production_only:
            modes = ['lora_nonzero', 'lora_fused_nonzero']
        execution_paths = ['eager'] + (['compiled'] if args.compile else []) + (['cudagraph'] if args.cudagraphs else [])
        for execution in execution_paths:
            compiled = execution != 'eager'
            cudagraphs = execution == 'cudagraph'
            for mode in modes:
                net = copy.deepcopy(template)
                if mode == 'frozen':
                    net.requires_grad_(False)
                zero_adapters = not mode.endswith('nonzero')
                if mode.startswith('lora'):
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == 'cuda' else []):
                        torch.manual_seed(118)
                        inject_lora(net, rank=8, alpha=8, targets=TARGETS, base_dtype=args.base_dtype,
                                    fused_qkv=mode == 'lora_fused_nonzero')
                        if not zero_adapters:
                            for layer in net.modules():
                                if isinstance(layer, LoRALinear):
                                    nn.init.normal_(layer.lora_B, std=.03)
                net.configure(compiled, args.checkpoint, compile_mode='block_cudagraph' if cudagraphs else 'block')
                for workload in (('scoring', 'rollout') if args.rollout else ('scoring',)):
                    for gradients in (False, True):
                        label = f'{execution}/{mode}/{workload}/{"grad" if gradients else "no_grad"}'
                        first = None
                        for iteration in range(args.iterations):
                            if cudagraphs:
                                torch.compiler.cudagraph_mark_step_begin()
                            result = run(net, source, workload, gradients, zero_adapters)
                            if first is None:
                                first = result
                            else:
                                repeat = differences(result, first, args.rtol, args.atol)
                                passed &= all(item['within_tolerance'] for item in repeat.values())
                                print(json.dumps(dict(case=label, iteration=iteration, repeat_difference=repeat)), flush=True)
                        results[label] = result
                        reference_mode = 'dense' if zero_adapters else 'lora_nonzero'
                        references = [f'eager/{reference_mode}/{workload}/{"grad" if gradients else "no_grad"}']
                        if compiled and mode == 'lora':
                            references.append(f'{execution}/frozen/{workload}/{"grad" if gradients else "no_grad"}')
                            references.append(f'eager/lora/{workload}/{"grad" if gradients else "no_grad"}')
                        if compiled and mode == 'lora_fused_nonzero':
                            references.append(f'{execution}/lora_nonzero/{workload}/{"grad" if gradients else "no_grad"}')
                        if cudagraphs:
                            references.append(f'compiled/{mode}/{workload}/{"grad" if gradients else "no_grad"}')
                        for reference in references:
                            diff = differences(result, results[reference], args.rtol, args.atol)
                            # Eager versus compiled BF16 rounding is diagnostic; compare
                            # graph replay and LoRA fusion against the same compiled math.
                            diagnostic = compiled and reference.startswith('eager/')
                            if not diagnostic:
                                passed &= all(item['aggregate_within_tolerance'] for item in diff.values())
                            print(json.dumps(dict(case=label, reference=reference, diagnostic=diagnostic, difference=diff)), flush=True)
                del net
    print(json.dumps(dict(checks_passed=passed, rtol=args.rtol, atol=args.atol,
                          cudagraph_skips=torch._dynamo.utils.counters['inductor']['cudagraph_skips'])), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
