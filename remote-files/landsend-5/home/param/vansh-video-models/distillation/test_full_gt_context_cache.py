"""Real tiny-model controls for differentiable fixed-history rollout caching."""

import copy
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from distillation.full_gt_context_cache import prefill_stabilizer_context, select_stabilizer_context
from distillation.full_gt_dmd import FullGTStabilizer
from distillation.test_joint_layout import clone_rows, compare, rows_and_noise, small_net
from distillation.test_rollout import attention_cpu


def inputs(batch=2, explicit_mask=True):
    rows, noise = rows_and_noise(batch=batch)
    for key in ('latents', 'pose'):
        rows[key] = rows[key][:, :, :8].detach().clone().requires_grad_(True)
    rows['ofi'] = rows['ofi'][:, :8].clone()
    rows['ofi'][0, 2] = -1
    if explicit_mask:
        choices = torch.tensor([[True, False, True, True, False, True],
                                [False, True, True, False, True, True]])
        rows['history_keep'] = choices.repeat((batch + 1) // 2, 1)[:batch].clone()
    noise = noise[:, :, :8].detach().clone().requires_grad_(True)
    return rows, noise


class FullGTContextCacheTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('distillation.bimodal_net.flash_attn_func', attention_cpu))
        stack.enter_context(patch('models.pose.audio.flash_attn_func', attention_cpu))
        # Match the existing real-model cache algebra tests: isolate cache and
        # gradient correctness from BF16 reduction order on CPU.
        stack.enter_context(patch.object(torch.Tensor, 'bfloat16', lambda tensor, *a, **kw: tensor.float()))

    def compare_gradients(self, original, cached, rows, copied, noise=None, copied_noise=None):
        for (name, parameter), (other_name, optimized) in zip(original.named_parameters(), cached.named_parameters()):
            self.assertEqual(name, other_name)
            with self.subTest(parameter=name):
                compare(self, parameter.grad, optimized.grad, 3e-7, 3e-4)
        for name in rows:
            with self.subTest(input=name):
                compare(self, rows[name].grad, copied[name].grad, 3e-7, 3e-4)
        if noise is not None:
            compare(self, noise.grad, copied_noise.grad, 3e-7, 3e-4)
        self.assertGreater(float(copied['refs'].grad.abs().sum()), 0.)
        self.assertGreater(float(cached.ref_patch_embedding.weight.grad.abs().sum()), 0.)
        self.assertIsNone(cached._references)
        self.assertIsNone(cached._reference_cache)

    def test_rollout_outputs_all_parameter_reference_and_noise_gradients(self):
        for trim in (False, True):
            for checkpoint in ('none', 'block'):
                with self.subTest(trim=trim, checkpoint=checkpoint):
                    torch.manual_seed(1741)
                    original = small_net()
                    original.configure(False, checkpoint)
                    cached = copy.deepcopy(original)
                    rows, noise = inputs()
                    copied = clone_rows(rows)
                    copied_noise = noise.detach().clone().requires_grad_(True)
                    reference = FullGTStabilizer(trim=trim, cache_context=False)
                    optimized = FullGTStabilizer(trim=trim)
                    _, context = reference.prepare_context(rows, noise)
                    _, copied_context = optimized.prepare_context(copied, copied_noise)
                    retained = {name: value.clone() for name, value in copied_context.items()}
                    expected_velocities, actual_velocities = [], []
                    rng = torch.get_rng_state().clone()
                    expected, returned_context = reference.generate(original, rows, noise, context,
                        observe_velocity=lambda value: expected_velocities.append(value.clone()))
                    actual, returned_copied = optimized.generate(cached, copied, copied_noise, copied_context,
                        observe_velocity=lambda value: actual_velocities.append(value.clone()))
                    compare(self, expected, actual, 2e-6, 2e-5)
                    self.assertEqual(len(actual_velocities), 8)
                    for a, b in zip(expected_velocities, actual_velocities):
                        compare(self, a, b, 2e-6, 2e-5)
                        self.assertFalse(b.requires_grad)
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                    self.assertIs(returned_context, context)
                    self.assertIs(returned_copied, copied_context)
                    self.assertEqual(set(copied_context), set(retained))
                    for name in retained:
                        torch.testing.assert_close(copied_context[name], retained[name], atol=0, rtol=0)
                    expected.square().mean().backward()
                    actual.square().mean().backward()
                    self.compare_gradients(original, cached, rows, copied, noise, copied_noise)
                    self.assertEqual(copied_noise.grad[:, :, :-1].count_nonzero().item(), 0)
                    self.assertGreater(float(copied_noise.grad[:, :, -1].abs().sum()), 0.)

    def test_mixed_band_raw_key_selection_preserves_relative_rope_and_gradients(self):
        torch.manual_seed(2137)
        original = small_net()
        original.configure(False, 'block')
        cached = copy.deepcopy(original)
        rows, noise = inputs(batch=4)
        copied = clone_rows(rows)
        copied_noise = noise.detach().clone().requires_grad_(True)
        rollout = FullGTStabilizer(trim=False)
        _, context = rollout.prepare_context(rows, noise)
        _, copied_context = rollout.prepare_context(copied, copied_noise)
        bands = torch.tensor([0, 1, 2, 3])
        times = torch.tensor([[.31], [.52], [.73], [.94]])
        with original.references(rows['refs'], rows['ref_keep']):
            expected = rollout._forward(original, noise[:, :, -1:], times, bands,
                context, original.audio_tokens(rows['audio']), 4, rows['drop'])
        with cached.references(copied['refs'], copied['ref_keep']):
            fixed = prefill_stabilizer_context(cached, copied_context, rollout.history_t)
            self.assertTrue(all(key.requires_grad and value.requires_grad for key, value in fixed))
            actual = rollout._forward(cached, copied_noise[:, :, -1:], times, bands,
                copied_context, cached.audio_tokens(copied['audio']), 4, copied['drop'],
                select_stabilizer_context(fixed, bands))
        compare(self, expected, actual, 2e-6, 2e-5)
        expected.square().mean().backward()
        actual.square().mean().backward()
        self.compare_gradients(original, cached, rows, copied, noise, copied_noise)

    def test_sampled_dropout_and_noise_preserve_rng_and_context(self):
        torch.manual_seed(3247)
        net = small_net()
        rows, _ = inputs(explicit_mask=False)
        reference = FullGTStabilizer(trim=False, cache_context=False)
        optimized = FullGTStabilizer(trim=False)
        before = torch.get_rng_state().clone()
        with torch.no_grad():
            expected, context = reference.generate(net, rows)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(before)
            actual, cached_context = optimized.generate(net, rows)
        compare(self, expected, actual, 2e-6, 2e-5)
        self.assertTrue(torch.equal(torch.get_rng_state(), after))
        self.assertFalse(cached_context['keep'][0, 1])  # source-invalid identity
        for name in context:
            torch.testing.assert_close(cached_context[name], context[name], atol=0, rtol=0)

    def test_fixed_seven_frame_prefill_once_and_scoring_stays_inline(self):
        torch.manual_seed(4337)
        net = small_net()
        rows, noise = inputs()
        for trim in (False, True):
            with self.subTest(trim=trim):
                rollout = FullGTStabilizer(trim=trim)
                _, context = rollout.prepare_context(rows, noise)
                calls = []
                forward = net.forward_pass
                def record(frames, plan, ext, audio, audio_mask, heads, **kwargs):
                    calls.append((len(frames), heads, kwargs.get('keep', ()), ext is None))
                    return forward(frames, plan, ext, audio, audio_mask, heads, **kwargs)
                with patch.object(net, 'forward_pass', record), torch.no_grad():
                    rollout.generate(net, rows, noise, context)
                    expected_present = [4, 4, 3, 3, 2, 2, 1, 1] if trim else [4] * 8
                    self.assertEqual(calls[0], (7, 0, tuple(range(7)), True))
                    self.assertEqual(calls[1:], [(n, n, (), False) for n in expected_present])
                    # Counts exclude reference prefill, which is identical in
                    # both implementations and lives below forward_pass().
                    frame_rows = rows['latents'].shape[0] * sum(call[0] for call in calls)
                    self.assertEqual(frame_rows, 2 * (27 if trim else 39))
                    calls.clear()
                    rollout.score_pass(net, rows, noise[:, :, -1:], torch.full((2, 1), .6), context)
                    self.assertEqual(calls, [(8, 4, (), True)])

    def test_cache_is_rebuilt_after_context_changes_and_exceptions(self):
        torch.manual_seed(5449)
        net = small_net()
        rows, noise = inputs()
        rollout = FullGTStabilizer(trim=False)
        _, context = rollout.prepare_context(rows, noise)
        with patch('distillation.full_gt_dmd.prefill_stabilizer_context',
                   wraps=prefill_stabilizer_context) as prefill, torch.no_grad():
            first, _ = rollout.generate(net, rows, noise, context)
            changed = {name: value.clone() for name, value in context.items()}
            changed['sink'].add_(1.)
            second, _ = rollout.generate(net, rows, noise, changed)
            self.assertEqual(prefill.call_count, 2)
            self.assertGreater(float((first - second).abs().max()), 1e-6)
            def fail(value):
                raise RuntimeError('observer failed')
            with self.assertRaisesRegex(RuntimeError, 'observer failed'):
                rollout.generate(net, rows, noise, context, observe_velocity=fail)
        self.assertIsNone(net._references)
        self.assertIsNone(net._reference_cache)


if __name__ == '__main__':
    unittest.main()
