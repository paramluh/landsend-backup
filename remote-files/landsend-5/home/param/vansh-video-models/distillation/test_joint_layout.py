"""Algebra, gradient, and layout checks for cached nine-reference joint passes."""

import copy
from contextlib import ExitStack
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn

from distillation.bimodal_net import BimodalNet, checkpointed
from distillation.joint_dmd import generate_resolver, generate_stabilizer, score_stabilizer
from distillation.test_rollout import attention_cpu
from models.pose.audio import AudioAdapter
from models.wan.model import WanModel


class TokenEncoder(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.null_token = nn.Parameter(torch.randn(1, 1, 1, dimension))
        self.no_audio_tokens = nn.Parameter(torch.randn(1, 1, 2, dimension))

    def forward(self, tokens):
        return tokens


def small_net(device='cpu'):
    dim = 32 if device == 'cuda' else 12
    transformer = WanModel(patch_size=(1, 1, 1), in_dim=2, out_dim=2, dim=dim, ffn_dim=2*dim,
                           freq_dim=8, num_heads=2, num_layers=2).to(device)
    if device == 'cuda':
        transformer.bfloat16()
    adapter = AudioAdapter(hidden_dim=dim, num_attn_heads=2, num_layers=2,
                           injection_offset=0, injection_interval=1, initialization='random').to(device)
    return BimodalNet(transformer, TokenEncoder(dim).to(device), adapter, nn.Conv3d(2, dim, 1).to(device),
                      nn.Parameter(torch.randn(dim, device=device)), (1, 2), 1, num_refs=9,
                      ref_patch_embedding=copy.deepcopy(transformer.patch_embedding)).to(device)


def rows_and_noise(device='cpu', batch=2):
    dim = 32 if device == 'cuda' else 12
    draw = lambda *shape: torch.randn(*shape, device=device, requires_grad=True)
    rows = dict(latents=draw(batch, 2, 11, 1, 2), pose=draw(batch, 2, 11, 1, 2),
                audio=draw(batch, 12, dim), refs=draw(batch, 2, 9, 1, 2),
                ref_keep=torch.ones(batch, 9, dtype=torch.bool, device=device),
                ofi=torch.arange(11, device=device)[None].expand(batch, -1).clone(),
                drop=torch.arange(batch, device=device).bool())
    rows['ref_keep'][0, [1, 7]] = False
    rows['ofi'][0, 5] = -1
    return rows, draw(batch, 2, 10, 1, 2)


def clone_rows(rows):
    return {key: value.detach().clone().requires_grad_(value.requires_grad) for key, value in rows.items()}


def compare(test, original, optimized, atol, rtol):
    if original is None:
        test.assertTrue(optimized is None or not optimized.count_nonzero().item())
    else:
        test.assertIsNotNone(optimized)
        torch.testing.assert_close(optimized.float(), original.float(), atol=atol, rtol=rtol)


class JointLayoutTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('distillation.bimodal_net.flash_attn_func', attention_cpu))
        self.stack.enter_context(patch('models.pose.audio.flash_attn_func', attention_cpu))
        # Isolate the cache/autograd algebra from BF16 summation order.
        self.stack.enter_context(patch.object(torch.Tensor, 'bfloat16', lambda tensor, *a, **kw: tensor.float()))

    def test_rollout_parameter_and_input_gradients(self):
        for checkpoint in ('none', 'block', 'call'):
            for prefix in (3, 6):
                with self.subTest(checkpoint=checkpoint, prefix=prefix):
                    torch.manual_seed(423)
                    reference = small_net()
                    cached = copy.deepcopy(reference)
                    reference.cache_references = False
                    reference.configure(False, checkpoint)
                    cached.configure(False, checkpoint)
                    rows, noise = rows_and_noise()
                    copied = clone_rows(rows)
                    copied_noise = noise.detach().clone().requires_grad_(True)
                    keep = torch.tensor([[True, False, True, False, True, True],
                                         [True, True, False, True, False, True]])
                    original_noise = noise[:, :, -prefix-4:]
                    actual_noise = copied_noise[:, :, -prefix-4:]
                    expected, expected_context = generate_stabilizer(reference, rows, noise=original_noise,
                        history_keep=keep, prefix_frames=prefix, static_layout=False)
                    actual, actual_context = generate_stabilizer(cached, copied, noise=actual_noise,
                        history_keep=keep, prefix_frames=prefix)
                    compare(self, expected, actual, 2e-6, 2e-5)
                    compare(self, expected_context['history'], actual_context['history'], 2e-6, 2e-5)
                    expected[:, :, -1].square().mean().backward()
                    actual[:, :, -1].square().mean().backward()
                    for (name, original), (_, optimized) in zip(reference.named_parameters(), cached.named_parameters()):
                        with self.subTest(parameter=name):
                            compare(self, original.grad, optimized.grad, 2e-7, 2e-4)
                    for name in rows:
                        compare(self, rows[name].grad, copied[name].grad, 2e-7, 2e-4)
                    compare(self, noise.grad, copied_noise.grad, 2e-7, 2e-4)
                    self.assertEqual(noise.grad[:, :, :prefix].count_nonzero().item(), 0)
                    for index in range(6, 10):
                        self.assertGreater(copied_noise.grad[:, :, index].abs().sum().item(), 0)
                    self.assertIsNone(cached._reference_cache)

    def test_resolver_and_guided_scoring_cache_gradients(self):
        for objective in ('resolver', 'guided_stabilizer'):
            with self.subTest(objective=objective):
                torch.manual_seed(657)
                reference = small_net()
                cached = copy.deepcopy(reference)
                reference.cache_references = False
                rows, noise = rows_and_noise()
                copied = clone_rows(rows)
                if objective == 'resolver':
                    expected = generate_resolver(reference, rows, noise=noise[:, :, -4:])
                    actual = generate_resolver(cached, copied, noise=noise.detach()[:, :, -4:])
                else:
                    context = dict(history=rows['latents'].detach()[:, :, 4:7],
                                   keep=torch.tensor([[False, True, True], [True, False, True]]))
                    times = torch.tensor([[.3, .5, .7, .9]]).expand(2, -1)
                    expected = score_stabilizer(reference, rows, noise[:, :, -4:], times, context, 2.5)
                    actual = score_stabilizer(cached, copied, noise.detach()[:, :, -4:], times, context, 2.5)
                compare(self, expected, actual, 2e-6, 2e-5)
                expected.square().mean().backward()
                actual.square().mean().backward()
                for (name, original), (_, optimized) in zip(reference.named_parameters(), cached.named_parameters()):
                    with self.subTest(parameter=name):
                        compare(self, original.grad, optimized.grad, 2e-7, 2e-4)
                for name in rows:
                    compare(self, rows[name].grad, copied[name].grad, 2e-7, 2e-4)

    def test_static_rollout_has_two_present_layouts(self):
        torch.manual_seed(7)
        net = small_net()
        rows, noise = rows_and_noise()
        layouts = set()
        original = net.forward_pass

        def record(frames, plan, ext, audio, audio_mask, heads, keep=(), **kwargs):
            if heads:
                layouts.add((len(frames), heads, tuple(keep), repr(plan), tuple(len(e) for e in ext)))
            return original(frames, plan, ext, audio, audio_mask, heads, keep, **kwargs)

        net.forward_pass = record
        generate_stabilizer(net, rows, noise=noise, history_keep=torch.zeros(2, 6, dtype=torch.bool))
        self.assertEqual(len(layouts), 2)  # detached prefix with K/V, targets without K/V
        self.assertEqual({layout[0] for layout in layouts}, {4})

    def test_undropped_extended_evaluation_and_nested_reference_scope(self):
        torch.manual_seed(91)
        reference = small_net()
        cached = copy.deepcopy(reference)
        reference.cache_references = False
        rows, _ = rows_and_noise()
        rows['latents'] = torch.randn(2, 2, 13, 1, 2)
        rows['pose'] = torch.randn_like(rows['latents'])
        rows['ofi'] = torch.arange(13)[None].expand(2, -1)
        rows['audio'] = torch.randn(2, 14, 12)
        noise, perturbation = torch.randn(2, 2, 12, 1, 2), torch.randn(2, 2, 3, 1, 2)
        with torch.no_grad():
            expected, context = generate_stabilizer(reference, rows, noise=noise, num_targets=6,
                history_perturbation=perturbation, static_layout=False)
            actual, actual_context = generate_stabilizer(cached, rows, noise=noise, num_targets=6,
                history_perturbation=perturbation)
            compare(self, expected, actual, 2e-6, 2e-5)
            compare(self, context['history'], actual_context['history'], 2e-6, 2e-5)
            with cached.references(rows['refs'], rows['ref_keep']):
                outer = cached._reference_cache
                with cached.references(rows['refs'] * 2, rows['ref_keep']):
                    self.assertIsNot(cached._reference_cache, outer)
                self.assertIs(cached._reference_cache, outer)
            self.assertIsNone(cached._reference_cache)

    def test_checkpoint_releases_activation_after_backward_with_output_alive(self):
        leaf = torch.randn(5, requires_grad=True)
        activation = leaf*2
        reference = weakref.ref(activation)
        output, = checkpointed(lambda value: (value.square(), None, None), activation)
        del activation
        self.assertIsNotNone(reference())
        output.sum().backward()
        self.assertIsNone(reference())
        torch.testing.assert_close(leaf.grad, 8*leaf)


@unittest.skipUnless(torch.cuda.is_available(), 'BF16 reference-cache parity requires CUDA')
class CudaJointLayoutTests(unittest.TestCase):
    def test_bf16_rollout_and_gradients(self):
        for prefix in (3, 6):
            with self.subTest(prefix=prefix):
                torch.manual_seed(423)
                reference = small_net('cuda')
                cached = copy.deepcopy(reference)
                reference.cache_references = False
                reference.configure(False, 'block')
                cached.configure(False, 'block')
                rows, noise = rows_and_noise('cuda')
                copied = clone_rows(rows)
                copied_noise = noise.detach().clone().requires_grad_(True)
                keep = torch.tensor([[True, False, True, False, True, True],
                                     [True, True, False, True, False, True]], device='cuda')
                expected, expected_context = generate_stabilizer(reference, rows, noise=noise[:, :, -prefix-4:],
                    history_keep=keep, prefix_frames=prefix, static_layout=False)
                actual, actual_context = generate_stabilizer(cached, copied, noise=copied_noise[:, :, -prefix-4:],
                    history_keep=keep, prefix_frames=prefix)
                compare(self, expected, actual, 5e-4, 5e-4)
                compare(self, expected_context['history'], actual_context['history'], 5e-4, 5e-4)
                expected[:, :, -1].square().mean().backward()
                actual[:, :, -1].square().mean().backward()
                pairs = [(name, parameter.grad, dict(cached.named_parameters())[name].grad)
                         for name, parameter in reference.named_parameters()]
                pairs += [(name, value.grad, copied[name].grad) for name, value in rows.items() if value.requires_grad]
                pairs.append(('noise', noise.grad, copied_noise.grad))
                for name, original, optimized in pairs:
                    with self.subTest(parameter=name):
                        if original is None:
                            compare(self, original, optimized, 0, 0)
                        else:
                            # Shared cached K/V sum BF16 gradients in a different
                            # order; the FP32 tests separately check the algebra.
                            delta = (optimized.float()-original.float()).norm()
                            self.assertLessEqual(delta.item(), .025*original.float().norm().item()+2e-7)


if __name__ == '__main__':
    unittest.main()
