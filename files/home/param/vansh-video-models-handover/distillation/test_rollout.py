import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from distillation.bimodal_dmd import UniformRollout, UniformStabilizerRollout
from distillation.bimodal_net import BimodalNet, checkpointed
from distillation.lora import LoRALinear, inject_lora
from models.pose.audio import AudioAdapter
from models.wan.model import WanModel


def attention_cpu(q, k, v):
    output = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float())
    return output.transpose(1, 2).to(q.dtype)


def small_net():
    transformer = WanModel(patch_size=(1, 1, 1), in_dim=2, out_dim=2, dim=12, ffn_dim=24,
                           freq_dim=8, num_heads=2, num_layers=2)
    encoder = nn.Module()
    encoder.null_token = nn.Parameter(torch.randn(1, 1, 1, 12))
    encoder.no_audio_tokens = nn.Parameter(torch.randn(1, 1, 2, 12))
    adapter = AudioAdapter(hidden_dim=12, num_attn_heads=2, num_layers=2,
                           injection_offset=0, injection_interval=1, initialization='random')
    return BimodalNet(transformer, encoder, adapter, nn.Conv3d(2, 12, 1), nn.Parameter(torch.randn(12)), (1, 2), 1)


class RolloutTests(unittest.TestCase):
    def test_compact_stabilizer_outputs_parameter_and_input_gradients(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        for mode in ('dense', 'lora', 'lora_fused'):
            for checkpoint, skip in [('none', ()), ('block', ()), ('block', (0,)), ('block', (1,))]:
                with self.subTest(mode=mode, checkpoint=checkpoint, skip=skip):
                    self.compare_rollouts(mode, checkpoint, skip)

    def compare_rollouts(self, mode, checkpoint, skip):
        torch.manual_seed(117)
        reference = small_net()
        if mode != 'dense':
            inject_lora(reference, rank=2, alpha=2, targets=[r'transformer\.blocks\..*(?:q|k|v|o|ffn\.[02])',
                                                          r'audio_adapter\..*(?:q_proj|kv_proj|out_proj)'])
            for layer in reference.modules():
                if isinstance(layer, LoRALinear):
                    nn.init.normal_(layer.lora_B, std=.03)
        compact = copy.deepcopy(reference)
        if mode == 'lora_fused':
            for block in compact.transformer.blocks:
                block.self_attn.lora_fused_qkv = True
        reference.configure(False, checkpoint)
        compact.configure(False, checkpoint, checkpoint_skip_blocks=skip)
        inputs = [torch.randn(2, 2, 8, 1, 2, requires_grad=True) for _ in range(3)]
        inputs += [torch.randn(2, 8, 12, requires_grad=True)]
        copies = [x.detach().clone().requires_grad_(True) for x in inputs]
        renoise = torch.randn_like(inputs[0])
        shapes = []
        forward_pass = compact.forward_pass

        def record(frames, plan, ext, audio, audio_mask, heads, keep=()):
            if len(frames) > 1:
                shapes.append((len(frames), heads, tuple(len(layer) for layer in ext), tuple(keep),
                               tuple(audio.shape), tuple(audio_mask.shape), copy.deepcopy(plan)))
            return forward_pass(frames, plan, ext, audio, audio_mask, heads, keep)

        compact.forward_pass = record

        def run(net, values, rollout):
            latents, pose, noise, tokens = values
            rows = dict(latents=latents, pose=pose, ofi=torch.arange(8)[None].expand(2, -1),
                        drop=torch.tensor([False, True]))
            y, resolved = rollout(net, rows, tokens, noise, renoise, stab_only=True)
            self.assertIsNone(resolved)
            y.square().mean().backward()
            return y

        with patch('distillation.bimodal_net.flash_attn_func', attention_cpu), \
                patch('models.pose.audio.flash_attn_func', attention_cpu):
            expected = run(reference, inputs, UniformRollout([2] * 4, 2, .2, .4))
            with patch('distillation.bimodal_net.checkpointed', wraps=checkpointed) as checkpoint_calls:
                actual = run(compact, copies, UniformStabilizerRollout([2] * 4, 2, .2, .4))
            self.assertEqual(checkpoint_calls.call_count, (2 - len(skip)) * 16 if checkpoint == 'block' else 0)
        self.assertEqual(tuple(actual.shape), (2, 2, 4, 1, 2))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        for (name, parameter), (_, ref) in zip(compact.named_parameters(), reference.named_parameters()):
            if ref.grad is None:
                self.assertIsNone(parameter.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, ref.grad, rtol=3e-3, atol=3e-5, msg=name)
        for original, copied in zip(inputs, copies):
            torch.testing.assert_close(copied.grad, original.grad, rtol=3e-3, atol=3e-5)
        self.assertEqual(len(shapes), 14)
        self.assertTrue(all(shape == shapes[0] for shape in shapes))
        self.assertEqual(shapes[0][:4], (5, 4, (6, 6), (4,)))


if __name__ == '__main__':
    unittest.main()
