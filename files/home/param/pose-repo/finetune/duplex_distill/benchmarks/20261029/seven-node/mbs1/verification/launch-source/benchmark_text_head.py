"""Isolated synthetic text-head benchmark; never changes training code.

Run CPU validation first with --cpu-only. CUDA runs must be explicitly assigned
an otherwise idle GPU using CUDA_VISIBLE_DEVICES. Timings include the actual
custom-autograd forward (which precomputes gradients) and backward, but exclude
input generation, correctness comparisons and CPU transfers.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import statistics
import time

import torch


def load_variants(path):
    spec = importlib.util.spec_from_file_location('text_head_benchmark_losses', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(path.read_text())

    class DirectAccumulation(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'add_'
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in ('dw', 'dr')
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Call)
                    and isinstance(node.args[0].func, ast.Name) and node.args[0].func.id == '_mm_accum'):
                self.count += 1
                return ast.copy_location(ast.Call(
                    func=ast.Name(id='_direct_addmm_', ctx=ast.Load()),
                    args=[node.func.value, *node.args[0].args], keywords=[]), node)
            return node

    # The isolated namespace makes the existing public wrapper resolve our
    # modified custom autograd class without patching the production module.
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in ('_ChunkedTextKL', 'factorized_text_kl_from_hidden')]
    transform = DirectAccumulation()
    direct_tree = ast.fix_missing_locations(transform.visit(ast.Module(body=selected, type_ignores=[])))
    if transform.count != 2:
        raise RuntimeError(f'Expected exactly two gradient accumulation replacements, got {transform.count}')
    namespace = dict(vars(module))
    namespace['_direct_addmm_'] = direct_addmm_
    exec(compile(direct_tree, str(path) + ':benchmark-direct-addmm', 'exec'), namespace)
    return {'baseline': module.factorized_text_kl_from_hidden,
            'direct_addmm': namespace['factorized_text_kl_from_hidden']}


def direct_addmm_(destination, a, b, dtype):
    if a.is_cuda and a.dtype in (torch.float16, torch.bfloat16) and dtype == torch.float32:
        return torch.addmm(destination, a, b, out_dtype=torch.float32, out=destination)
    return torch.addmm(destination, a.to(dtype), b.to(dtype), out=destination)


def cpu_validate(variants):
    torch.manual_seed(9481)
    torch.set_num_threads(1)
    specs = (15, 17, 18)
    tensors = [torch.randn(*shape, dtype=torch.float64, requires_grad=True)
               for shape in ((11, 7), (19, 7), (3, 7), (11, 9), (19, 9), (3, 9))]
    labels = torch.tensor([0, 1, 2, 15, 17, 18, 3, 8, -100, 6, 11])
    weights = torch.arange(1., 12., dtype=torch.float64) / 110
    weights[8] = 0
    sx, sw, sr, tx, tw, tr = tensors
    student = sx @ sw.t()
    teacher = tx.detach() @ tw.detach().t()
    student[:, list(specs)] = sx @ sr.t()
    teacher[:, list(specs)] = tx.detach() @ tr.detach().t()
    slp, tlp = student.log_softmax(-1), teacher.log_softmax(-1)
    full = (tlp.exp() * (tlp - slp)).sum(-1)
    slw, tlw = student[:, :14].log_softmax(-1), teacher[:, :14].log_softmax(-1)
    lexical = (tlw.exp() * (tlw - slw)).sum(-1)
    mass = tlp[:, :14].exp().sum(-1)
    checks = []
    for lexical_weight in (1., 5.):
        ref = ((full + (lexical_weight - 1) * mass * lexical) * weights).sum()
        expected_grads = torch.autograd.grad(ref * .37, tensors[:3], retain_graph=True)
        for name, function in variants.items():
            for chunk in (2, 4):
                loss, stats = function(*tensors, labels, weights, specs, word_end=14,
                                       lexical_weight=lexical_weight, chunk_size=chunk)
                gradients = torch.autograd.grad(loss * .37, tensors[:3])
                torch.testing.assert_close(loss, ref, rtol=1e-12, atol=1e-12)
                for got, expected in zip(gradients, expected_grads):
                    torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-12)
                assert all(t.grad is None for t in tensors[3:])
                assert all(not value.requires_grad for value in stats.values())
                checks.append({'variant': name, 'chunk': chunk, 'lexical_weight': lexical_weight, 'pass': True})
    return {'pass': True, 'dtype': 'float64', 'independent_autograd_reference': True,
            'teacher_and_diagnostic_gradients_absent': True, 'checks': checks}


@torch.no_grad()
def compare_gradients(got, expected_cpu):
    diff_sq = torch.zeros((), device='cuda', dtype=torch.float64)
    ref_sq = torch.zeros_like(diff_sq)
    max_diff = torch.zeros((), device='cuda')
    max_ref = torch.zeros_like(max_diff)
    # Compare every gradient element without materializing another full fp32
    # vocabulary-by-hidden matrix on the GPU.
    for start in range(0, got.shape[0], 1024):
        ref = expected_cpu[start:start + 1024].to('cuda', dtype=torch.float32)
        diff = got[start:start + 1024].float() - ref
        diff_sq += diff.square().sum(dtype=torch.float64)
        ref_sq += ref.square().sum(dtype=torch.float64)
        max_diff = torch.maximum(max_diff, diff.abs().amax())
        max_ref = torch.maximum(max_ref, ref.abs().amax())
    result = {'relative_l2': float((diff_sq / ref_sq.clamp_min(1e-30)).sqrt()),
              'max_absolute': float(max_diff),
              'max_error_over_reference_max': float(max_diff / max_ref.clamp_min(1e-30))}
    result['pass'] = result['relative_l2'] <= .02 and result['max_error_over_reference_max'] <= .10
    return result


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(path)


def gpu_benchmark(variants, args, result):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA benchmark requires an explicitly assigned GPU')
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device, dtype = 'cuda', torch.bfloat16
    scale = 1 / math.sqrt(args.width)
    tensors = [torch.randn(*shape, device=device, dtype=dtype)
               for shape in ((args.rows, args.width), (args.vocab, args.width), (3, args.width),
                             (args.rows, args.width), (args.vocab, args.width), (3, args.width))]
    for i in (1, 2, 4, 5):
        tensors[i].mul_(scale)
    for tensor in tensors[:3]:
        tensor.requires_grad_(True)
    targets = torch.randint(0, args.word_end, (args.rows,), device=device)
    specs = (151670, 151671, 151645)
    targets[::7], targets[1::7], targets[2::7] = specs
    weights = torch.full((args.rows,), 1 / args.rows, device=device)
    torch.cuda.synchronize()
    result.update(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                  cases=[], tolerances={'loss_rtol': .001, 'loss_atol': .00001,
                                       'gradient_relative_l2_max': .02, 'gradient_max_error_over_reference_max': .10})
    reference = None
    for name, function in variants.items():
        for chunk in args.chunks:
            case = {'variant': name, 'chunk': chunk}
            result['cases'].append(case)
            try:
                for tensor in tensors[:3]:
                    tensor.grad = None
                gc.collect()
                torch.cuda.empty_cache()
                # Warm up the exact case, then remove all output/gradient tensors.
                for _ in range(args.warmup):
                    loss, stats = function(*tensors, targets, weights, specs,
                                          word_end=args.word_end, chunk_size=chunk)
                    loss.backward()
                    del loss, stats
                    for tensor in tensors[:3]:
                        tensor.grad = None
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                case['initial_allocated_bytes'] = torch.cuda.memory_allocated()
                elapsed, forward_times, backward_times = [], [], []
                wall_start = time.perf_counter()
                for _ in range(args.iterations):
                    for tensor in tensors[:3]:
                        tensor.grad = None
                    start, midpoint, finish = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                    start.record()
                    loss, stats = function(*tensors, targets, weights, specs,
                                          word_end=args.word_end, chunk_size=chunk)
                    midpoint.record()
                    loss.backward()
                    finish.record()
                    finish.synchronize()
                    elapsed.append(start.elapsed_time(finish))
                    forward_times.append(start.elapsed_time(midpoint))
                    backward_times.append(midpoint.elapsed_time(finish))
                case.update(total_ms=elapsed, forward_ms=forward_times, backward_ms=backward_times,
                            median_ms=statistics.median(elapsed), mean_ms=statistics.mean(elapsed),
                            timed_loop_wall_seconds=time.perf_counter() - wall_start,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(), loss=float(loss))
                if reference is None:
                    reference = {'loss': float(loss),
                                 'grads': [tensor.grad.detach().cpu() for tensor in tensors[:3]],
                                 'stats': {key: value.detach().cpu() for key, value in stats.items()}}
                errors = [compare_gradients(tensor.grad, expected)
                          for tensor, expected in zip(tensors[:3], reference['grads'])]
                loss_pass = math.isclose(float(loss), reference['loss'], rel_tol=.001, abs_tol=.00001)
                case['correctness'] = {'loss_absolute_error': abs(float(loss) - reference['loss']),
                                       'loss_pass': loss_pass,
                                       'gradients': dict(zip(('hidden', 'weight', 'special_rows'), errors)),
                                       'pass': loss_pass and all(error['pass'] for error in errors)}
                del loss, stats
                if not case['correctness']['pass']:
                    raise AssertionError(f'Gradient/loss comparison failed: {case["correctness"]}')
                print(json.dumps(case), flush=True)
            except torch.cuda.OutOfMemoryError as exc:
                case['error'] = str(exc)
                case['oom'] = True
                print(json.dumps({'variant': name, 'chunk': chunk, 'oom': True}), flush=True)
            finally:
                for tensor in tensors[:3]:
                    tensor.grad = None
                gc.collect()
                torch.cuda.empty_cache()
                write_result(args.output, result)
    del tensors, reference
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result['final_allocated_bytes'] = torch.cuda.memory_allocated()
    result['complete'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--losses', type=Path, default=Path(__file__).with_name('losses.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--rows', type=int, default=7499)
    parser.add_argument('--width', type=int, default=5120)
    parser.add_argument('--vocab', type=int, default=151936)
    parser.add_argument('--word-end', type=int, default=151643)
    parser.add_argument('--chunks', type=int, nargs='+', default=[32, 128, 256, 512])
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--seed', type=int, default=71917)
    args = parser.parse_args()
    variants = load_variants(args.losses)
    result = {'benchmark': 'synthetic_production_shape_text_head', 'host': socket.gethostname(),
              'inputs': 'Deterministic synthetic Gaussian hidden states and scaled Gaussian head weights; random dataset labels',
              'shape': {'rows': args.rows, 'hidden_width': args.width, 'vocab': args.vocab},
              'dtype': 'bfloat16', 'seed': args.seed, 'warmup_iterations': args.warmup,
              'timed_iterations': args.iterations, 'includes': 'full custom-autograd forward and backward',
              'excludes': 'other model modules, FSDP communication, data loading, optimizer, correctness checks',
              'source_sha256': hashlib.sha256(args.losses.read_bytes()).hexdigest(),
              'cpu_validation': cpu_validate(variants)}
    write_result(args.output, result)
    print(json.dumps({'cpu_validation': result['cpu_validation']}), flush=True)
    if not args.cpu_only:
        gpu_benchmark(variants, args, result)
    write_result(args.output, result)


if __name__ == '__main__':
    main()
