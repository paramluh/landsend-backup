import copy
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch

from distillation.resolver_dmd import (
    GTConditionedStabilizer, gt_noised_input, one_step_resolve, resolve_gt_pass,
)
from distillation.test_rollout import attention_cpu, small_net


def independent_resolve(net, rows, x, times, audio):
    outputs = []
    for j in range(4):
        f = rows['latents'].shape[2]-4+j
        frames = [(x[:, :, j], rows['pose'][:, :, f], times[:, j], 3, True)]
        frames += [(rows['latents'][:, :, f-2+i], rows['pose'][:, :, f-2+i], 0., 2,
                    rows['ofi'][:, f-2+i] >= 0)
                   for i in range(2)]
        frames += [(rows['latents'][:, :, 0], None, 0., 0, True)]
        plan = [(0, [[('f', 3), ('r', 1, 1), ('r', 2, 2), ('f', 0)]]),
                (1, [[('f', 3), ('f', 1)], [('f', 3), ('f', 2)]]), (3, [[('f', 3)]])]
        output, _, _ = net.forward_pass(frames, plan, None, audio[:, j:j+1],
                                        torch.ones(x.shape[0], 1, dtype=torch.bool), 1)
        outputs.append(output)
    return torch.cat(outputs, dim=2)


class ResolverDMDTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(408)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def rows(self, requires_grad=False):
        return dict(latents=torch.randn(2, 2, 11, 1, 2, requires_grad=requires_grad),
                    pose=torch.randn(2, 2, 11, 1, 2, requires_grad=requires_grad),
                    ofi=torch.arange(11)[None].expand(2, -1),
                    drop=torch.tensor([False, True]))

    def test_packed_resolver_matches_independent_windows_and_all_gradients(self):
        for full_precision in [False, True]:
            with self.subTest(full_precision=full_precision):
                self.compare_packed_gradients(full_precision)

    def compare_packed_gradients(self, full_precision):
        packed = small_net()
        independent = copy.deepcopy(packed)
        packed.configure(False, 'none')
        independent.configure(False, 'none')
        rows = self.rows(True)
        rows['ofi'] = rows['ofi'].clone()
        rows['ofi'][0, 5] = -1
        reference_rows = {k: v.detach().clone().requires_grad_(v.requires_grad) for k,v in rows.items()}
        x = torch.randn(2, 2, 4, 1, 2, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        audio = torch.randn(2, 4, 2, 12, requires_grad=True)
        audio_ref = audio.detach().clone().requires_grad_(True)
        times = torch.rand(2, 4)*.2
        weight = torch.randn_like(x)
        # FP32 establishes mathematical gradient equivalence. The production
        # BF16 attention lane additionally checks bounded rounding from summing
        # shared-history gradients before rather than after their BF16 cast.
        precision = patch.object(torch.Tensor, 'bfloat16', lambda value: value) if full_precision else nullcontext()
        with precision, patch('distillation.bimodal_net.flash_attn_func', attention_cpu), \
             patch('models.pose.audio.flash_attn_func', attention_cpu):
            actual = resolve_gt_pass(packed, rows, x, times, audio, torch.ones(2, 4, dtype=torch.bool))
            expected = independent_resolve(independent, reference_rows, x_ref, times, audio_ref)
            (actual*weight).sum().backward()
            (expected*weight).sum().backward()
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
        for (name, parameter), (_, ref) in zip(packed.named_parameters(), independent.named_parameters()):
            if ref.grad is None:
                self.assertIsNone(parameter.grad, name)
            elif full_precision:
                torch.testing.assert_close(parameter.grad, ref.grad, rtol=2e-5, atol=5e-6, msg=name)
            else:
                relative = (parameter.grad-ref.grad).norm()/ref.grad.norm().clamp_min(1e-8)
                self.assertLess(float(relative), .005, name)
        for actual_input, reference_input in [(x, x_ref), (audio,audio_ref),
                                               (rows['latents'],reference_rows['latents']),
                                               (rows['pose'],reference_rows['pose'])]:
            if full_precision:
                torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=2e-5, atol=5e-6)
            else:
                relative = (actual_input.grad-reference_input.grad).norm()/reference_input.grad.norm().clamp_min(1e-8)
                self.assertLess(float(relative), .005)

    def test_dropped_history_ignores_latents_and_pose_and_matches_independent_windows(self):
        net = small_net(); net.configure(False, 'none')
        rows = self.rows(True)
        rows['ofi'] = rows['ofi'].clone()
        rows['ofi'][0, 5] = -1
        poisoned = {k: v.detach().clone() for k, v in rows.items()}
        poisoned['latents'][0, :, 5] = 1e6
        poisoned['pose'][0, :, 5] = -1e6
        x = torch.randn(2, 2, 4, 1, 2)
        audio = torch.randn(2, 4, 2, 12)
        times = torch.full((2, 4), .2)
        mask = torch.ones(2, 4, dtype=torch.bool)
        with patch('distillation.bimodal_net.flash_attn_func', attention_cpu), \
             patch('models.pose.audio.flash_attn_func', attention_cpu):
            actual = resolve_gt_pass(net, rows, x, times, audio, mask)
            altered = resolve_gt_pass(net, poisoned, x, times, audio, mask)
            reference = independent_resolve(net, rows, x, times, audio)
            actual.square().sum().backward()
        torch.testing.assert_close(actual, altered, rtol=0, atol=0)
        torch.testing.assert_close(actual, reference, rtol=3e-4, atol=3e-5)
        self.assertEqual(int(torch.count_nonzero(rows['latents'].grad[0, :, 5])), 0)
        self.assertEqual(int(torch.count_nonzero(rows['pose'].grad[0, :, 5])), 0)

    def test_generated_target_never_changes_other_targets_history(self):
        net = small_net()
        net.configure(False, 'none')
        rows = self.rows()
        x = torch.randn(2, 2, 4, 1, 2)
        changed = x.clone(); changed[:, :, 0] += 100
        audio = torch.randn(2, 4, 2, 12)
        times = torch.full((2,4), .2)
        mask = torch.ones(2,4,dtype=torch.bool)
        with torch.no_grad(), patch('distillation.bimodal_net.flash_attn_func', attention_cpu), \
             patch('models.pose.audio.flash_attn_func', attention_cpu):
            before = resolve_gt_pass(net, rows, x, times, audio, mask)
            after = resolve_gt_pass(net, rows, changed, times, audio, mask)
        torch.testing.assert_close(after[:,:,1:],before[:,:,1:],rtol=0,atol=0)
        self.assertFalse(torch.equal(before[:,:,0],after[:,:,0]))

    def test_one_step_has_correct_input_and_parameter_gradients(self):
        class LinearField:
            def __init__(self):
                self.scale = torch.tensor(.7, requires_grad=True)
            def frame_audio(self, tokens, ofi, drop):
                return torch.zeros(ofi.shape[0],ofi.shape[1],1,1)
            def forward_pass(self, frames, plan, ext, audio, mask, heads):
                return self.scale*torch.stack([f[0] for f in frames[:heads]],2),None,None
        net = LinearField(); rows = self.rows()
        noise = torch.randn_like(rows['latents'][:,:,7:])
        x = gt_noised_input(rows, noise).detach().requires_grad_(True)
        torch.testing.assert_close(x,.8*rows['latents'][:,:,7:]+.2*noise)
        y = one_step_resolve(net,rows,x,tokens=torch.empty(0))
        torch.testing.assert_close(y,x*(1-.2*net.scale))
        y.sum().backward()
        torch.testing.assert_close(x.grad,torch.full_like(x,1-.2*.7))
        torch.testing.assert_close(net.scale.grad,-.2*x.detach().sum())

    def test_onpolicy_generation_is_detached_and_requires_complete_gt_context(self):
        net = small_net();net.configure(False,'none')
        rows = self.rows(True);tokens=torch.randn(2,11,12,requires_grad=True)
        rollout=GTConditionedStabilizer()
        with patch('distillation.bimodal_net.flash_attn_func', attention_cpu), \
             patch('models.pose.audio.flash_attn_func', attention_cpu):
            y=rollout(net,rows,tokens,torch.randn_like(rows['latents']))
        self.assertEqual(y.shape,(2,2,4,1,2))
        self.assertFalse(y.requires_grad)
        self.assertTrue(torch.isfinite(y).all())
        short={**rows,'latents':rows['latents'][:,:,:8],'pose':rows['pose'][:,:,:8]}
        with self.assertRaisesRegex(ValueError,'six preceding'):
            rollout(net,short,tokens,torch.randn_like(short['latents']))


if __name__=='__main__':
    unittest.main()
