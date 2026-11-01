"""Teacher-only, whole-clip batched audio CFG for single-target full-GT DMD."""
import ast
import copy
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from distillation.full_gt_dmd import FullGTStabilizer
from distillation.test_full_gt_context_cache import inputs
from distillation.test_joint_layout import clone_rows, compare, small_net
from distillation.test_rollout import attention_cpu


def separate_queries(rollout, net, rows, x, times, context, scale):
    """Independent, pre-batching reference: two full audio-conditioning passes."""
    bands = ((times[:, 0] - rollout.history_t) / rollout.width - 1e-6).floor().long().clamp(0, 3)
    with net.references(rows['refs'], rows['ref_keep']):
        tokens = net.audio_tokens(rows['audio'])
        conditional = rollout._forward(net, x, times, bands, context, tokens, 4, context['drop'])
        null = rollout._forward(net, x, times, bands, context, tokens, 4,
                                torch.ones_like(context['drop']))
    return torch.where((bands == 3)[:, None, None, None, None],
                       null + scale * (conditional - null), conditional)


class FullGTGuidanceTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('distillation.bimodal_net.flash_attn_func', attention_cpu))
        stack.enter_context(patch('models.pose.audio.flash_attn_func', attention_cpu))
        stack.enter_context(patch.object(torch.Tensor, 'bfloat16', lambda value, *a, **kw: value.float()))

    def test_real_model_output_and_all_parameter_input_gradients_match_separate_passes(self):
        for checkpoint in ('none', 'block'):
            with self.subTest(checkpoint=checkpoint):
                torch.manual_seed(7241)
                original = small_net()
                original.configure(False, checkpoint)
                batched = copy.deepcopy(original)
                rows, noise = inputs(batch=6)
                rows['drop'].zero_()
                rows['drop'][5] = True  # Bad-audio semantics, if present, remain unchanged.
                copied = clone_rows(rows)
                x = noise[:, :, -1:].detach().clone().requires_grad_()
                other_x = x.detach().clone().requires_grad_()
                times = torch.tensor([[.3], [.4], [.6], [.8], [.81], [.99]])
                rollout = FullGTStabilizer()
                _, context = rollout.prepare_context(rows, noise)
                _, other_context = rollout.prepare_context(copied, noise.detach())
                before = torch.get_rng_state().clone()
                expected = separate_queries(rollout, original, rows, x, times, context, 2.5)
                actual = rollout.score_pass(batched, copied, other_x, times, other_context, 2.5)
                compare(self, expected, actual, 2e-6, 2e-5)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                expected.square().mean().backward()
                actual.square().mean().backward()
                for (name, parameter), (_, other) in zip(original.named_parameters(), batched.named_parameters()):
                    with self.subTest(parameter=name):
                        compare(self, parameter.grad, other.grad, 5e-7, 5e-4)
                for name in rows:
                    with self.subTest(input=name):
                        compare(self, rows[name].grad, copied[name].grad, 5e-7, 5e-4)
                compare(self, x.grad, other_x.grad, 5e-7, 5e-4)
                self.assertGreater(float(copied['refs'].grad.abs().sum()), 0.)
                self.assertIsNone(batched._reference_cache)
                self.assertIsNone(batched._references)

    def test_each_top_band_count_uses_one_pass_and_nulls_every_audio_window(self):
        torch.manual_seed(7251)
        net = small_net()
        rows, noise = inputs()
        rows['drop'].zero_()
        # Production Mimi latents are a transposed [B,T,D] view. Guidance
        # must preserve this encoder input layout and encode only B rows.
        rows['audio'] = rows['audio'].transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(rows['audio'].is_contiguous())
        rollout = FullGTStabilizer()
        _, context = rollout.prepare_context(rows, noise)
        for times, expected_nulls in (([.3, .8], 0), ([.8, .9], 1), ([.81, 1.], 2)):
            with self.subTest(times=times):
                audio_inputs = []
                original_audio = net.frame_audio
                def frame_audio(tokens, indices, drop):
                    audio = original_audio(tokens, indices, drop)
                    audio_inputs.append((indices.clone(), drop.clone(), audio.clone()))
                    return audio
                with patch.object(net, 'frame_audio', frame_audio), \
                        patch.object(net, 'audio_tokens', wraps=net.audio_tokens) as encode, \
                        patch.object(net, 'forward_pass', wraps=net.forward_pass) as forward, \
                        patch.object(net, '_prefill_references', wraps=net._prefill_references) as prefill, \
                        torch.no_grad():
                    actual = rollout.score_pass(net, rows, noise[:, :, -1:],
                        torch.tensor(times)[:, None], context, 2.5)
                self.assertEqual(forward.call_count, 1)
                self.assertEqual(prefill.call_count, 1)
                self.assertEqual(encode.call_count, 1)
                self.assertIs(encode.call_args.args[0], rows['audio'])
                self.assertEqual(encode.call_args.args[0].shape[0], 2)
                packed_refs, packed_keep = prefill.call_args.args
                indices, drop, audio = audio_inputs[0]
                self.assertEqual(tuple(drop.shape), (2 + expected_nulls,))
                self.assertEqual(drop.tolist(), [False, False] + [True] * expected_nulls)
                self.assertEqual(audio.shape[1], 4)
                torch.testing.assert_close(packed_refs[:2], rows['refs'], rtol=0, atol=0)
                self.assertEqual(actual.shape, noise[:, :, -1:].shape)
                selected = torch.tensor([t > .8 for t in times]).nonzero().flatten()
                torch.testing.assert_close(packed_refs[2:], rows['refs'][selected], rtol=0, atol=0)
                torch.testing.assert_close(packed_keep[2:], rows['ref_keep'][selected], rtol=0, atol=0)
                torch.testing.assert_close(indices[2:], indices[selected], rtol=0, atol=0)
                null = net.audio_encoder.no_audio_tokens.expand(expected_nulls, 4, -1, -1)
                torch.testing.assert_close(audio[2:], null, rtol=0, atol=0)
                self.assertFalse(context['drop'].any())  # No mutation of critic/student context.

    def test_trainer_routes_guidance_to_teacher_stabilizer_only(self):
        source = ast.parse((Path(__file__).resolve().parents[1] / 'learn_dmd_joint.py').read_text())
        score = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == 'score')
        rollout, resolver = SimpleNamespace(score_pass=Mock()), Mock()
        namespace = {'full_gt': True, 'full_gt_rollout': rollout, 'score_full_gt_resolver': resolver,
                     'config': {'teacher_guidance_scale': 2.5}}
        exec(compile(ast.Module(body=[score], type_ignores=[]), 'learn_dmd_joint.py', 'exec'), namespace)
        for teacher in (False, True):
            namespace['score']('net', 'rows', 'stabilizer', 'x', 'times', 'context', teacher)
            self.assertEqual(rollout.score_pass.call_args.kwargs['guidance_scale'], 2.5 if teacher else 1.)
            namespace['score']('net', 'rows', 'resolver', 'x', 'times', 'context', teacher)
            self.assertEqual(resolver.call_args.args, ('net', 'rows', 'x', 'times', 'context'))
            self.assertEqual(resolver.call_args.kwargs, {})


if __name__ == '__main__':
    unittest.main()
