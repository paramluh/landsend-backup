"""CPU numerical checks for the variable-size, cached stabilizer rollout."""

import copy
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch
from torch import nn

from distillation.bimodal_dmd import Rollout, UniformStabilizerRollout
from distillation.bimodal_net import checkpointed
from distillation.lora import LoRALinear, inject_lora
from distillation.test_rollout import attention_cpu, small_net


class TrimmedRolloutTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)

    def test_outputs_all_parameter_and_input_gradients(self):
        for fp32_attention in (False, True):
            for mode in ('dense', 'lora', 'lora_fused'):
                for checkpoint, skip in [('none', ()), ('block', ()), ('block', (0,)),
                                         ('block', (1,)), ('call', ())]:
                    with self.subTest(mode=mode, checkpoint=checkpoint, skip=skip, fp32_attention=fp32_attention):
                        self.compare_rollouts(mode, checkpoint, skip, fp32_attention=fp32_attention)

    def test_last_frame_reaches_all_previous_generated_frame_noise(self):
        for mode in ('dense', 'lora'):
            with self.subTest(mode=mode):
                self.compare_rollouts(mode, 'block', (0,), last_frame_only=True)

    def compare_rollouts(self, mode, checkpoint, skip, last_frame_only=False, fp32_attention=False):
        torch.manual_seed(233)
        reference = small_net()
        if mode != 'dense':
            inject_lora(reference, rank=2, alpha=2,
                        targets=[r'transformer\.blocks\..*(?:q|k|v|o|ffn\.[02])',
                                 r'audio_adapter\..*(?:q_proj|kv_proj|out_proj)'],
                        fused_qkv=mode == 'lora_fused')
            for layer in reference.modules():
                if isinstance(layer, LoRALinear):
                    nn.init.normal_(layer.lora_B, std=.03)
        trimmed = copy.deepcopy(reference)
        # Compare cached, selectively checkpointed passes against the static reference.
        reference.configure(False, checkpoint)
        trimmed.configure(False, checkpoint, checkpoint_skip_blocks=skip)
        inputs = [torch.randn(4, 2, 8, 1, 2, requires_grad=True) for _ in range(3)]
        inputs.append(torch.randn(4, 8, 12, requires_grad=True))
        copies = [value.detach().clone().requires_grad_(True) for value in inputs]
        renoise = torch.randn_like(inputs[0])
        records = {}

        def run(net, values, rollout, label):
            latents, pose, noise, tokens = values
            rows = dict(latents=latents, pose=pose, ofi=torch.arange(8)[None].expand(4, -1),
                        drop=torch.tensor([False, True, False, True]))
            records[label] = []
            original = net.forward_pass

            def record(frames, plan, ext, audio, audio_mask, heads, keep=()):
                result = original(frames, plan, ext, audio, audio_mask, heads, keep)
                # Check the cache itself remains in autograd, including sink and pad prefills.
                for tensor in result[1] + result[2]:
                    if tensor is not None:
                        self.assertTrue(tensor.requires_grad)
                records[label].append((len(frames), heads, tuple(keep),
                                       sum(bool(frame[4]) for frame in frames)))
                return result

            net.forward_pass = record
            output, resolved = rollout(net, rows, tokens, noise, renoise, stab_only=True)
            self.assertIsNone(resolved)
            selected = output[:, :, -1:] if last_frame_only else output
            weights = torch.linspace(.1, 1.3, selected.numel()).view_as(selected)
            (selected.square() * weights).mean().backward()
            return output

        # Retaining fp32 intermediates checks the cache/autograd algebra separately from BF16 accumulation
        # order: shared cached K/V gradients can accumulate differently from recomputed identical K/V.
        attention_precision = (patch.object(torch.Tensor, 'bfloat16', lambda tensor, *args, **kwargs: tensor.float())
                               if fp32_attention else nullcontext())
        with patch('distillation.bimodal_net.flash_attn_func', attention_cpu), \
                patch('models.pose.audio.flash_attn_func', attention_cpu), attention_precision:
            expected = run(reference, inputs, UniformStabilizerRollout([2] * 4, 2, .2, .4), 'uniform')
            with patch('distillation.bimodal_net.checkpointed', wraps=checkpointed) as calls:
                actual = run(trimmed, copies, Rollout([2] * 4, 2, .2, .4), 'trimmed')
            self.assertEqual(calls.call_count, (2 - len(skip)) * 16 if checkpoint == 'block' else 0)

        self.assertEqual(tuple(actual.shape), (4, 2, 4, 1, 2))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        reference_parameters, actual_parameters = dict(reference.named_parameters()), dict(trimmed.named_parameters())
        self.assertEqual(reference_parameters.keys(), actual_parameters.keys())
        tolerance = dict(rtol=1e-4, atol=1e-7) if fp32_attention else dict(rtol=3e-3, atol=8e-5)
        for name, parameter in actual_parameters.items():
            expected_grad = reference_parameters[name].grad
            if expected_grad is None:
                self.assertIsNone(parameter.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, expected_grad, **tolerance, msg=name)
        for label, original, copied in zip(('latents', 'pose', 'noise', 'audio'), inputs, copies):
            torch.testing.assert_close(copied.grad, original.grad, **tolerance, msg=label)
        if last_frame_only:
            # An accidentally detached inter-frame K/V cache would break these earlier-frame paths.
            for frame in range(4, 8):
                self.assertGreater(copies[2].grad[:, :, frame].abs().sum().item(), 0, f'noise frame {frame}')

        uniform, fused = records['uniform'][2:], records['trimmed'][2:]
        self.assertEqual(len(uniform), 14)
        self.assertEqual(len(fused), 14)
        self.assertEqual([entry[0] for entry in fused], [4, 4, 5, 4, 5, 4, 5, 4, 4, 3, 3, 2, 2, 1])
        self.assertEqual(sum(entry[0] for entry in uniform), 70)
        self.assertEqual(sum(entry[0] for entry in fused), 50)
        self.assertEqual(sum(entry[3] for entry in uniform), 56)
        self.assertEqual(sum(entry[3] for entry in fused), 50)
        self.assertEqual(sum(entry[1] for entry in fused), 44)
        self.assertEqual(sum(len(entry[2]) for entry in fused), 6)


if __name__ == '__main__':
    unittest.main()
