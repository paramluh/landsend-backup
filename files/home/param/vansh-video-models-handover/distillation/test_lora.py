import copy
import unittest

import torch
from torch import nn

from distillation.bimodal_net import checkpointed
from distillation.lora import LoRALinear, fused_lora_qkv, inject_lora, merged_state_dict


class LoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def model(self):
        return nn.Sequential(nn.Linear(5, 8), nn.GELU(), nn.Linear(8, 3))

    def test_zero_initialization_and_frozen_base(self):
        model = self.model()
        x = torch.randn(4, 5)
        expected = model(x).detach()
        base = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        self.assertEqual(inject_lora(model, rank=2, alpha=4, targets=[r'0|2']), ['0', '2'])
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual(sum(parameter.numel() for parameter in trainable), 48)
        optimizer = torch.optim.AdamW(trainable, lr=.05)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
            optimizer.step()
        for name, parameter in model.named_parameters():
            if name in base:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, base[name], rtol=0, atol=0)
            else:
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertFalse(torch.equal(model(x), expected))

    def test_merge_and_adapter_checkpoint(self):
        model = self.model()
        dense = copy.deepcopy(model)
        inject_lora(model, rank=2, alpha=4, targets=[r'0|2'])
        for layer in model.modules():
            if isinstance(layer, LoRALinear):
                nn.init.normal_(layer.lora_B, std=.1)
        x = torch.randn(4, 5, requires_grad=True)
        expected = model(x)
        expected_gradient, = torch.autograd.grad(expected.square().sum(), x)
        dense.load_state_dict(merged_state_dict(model), strict=True)
        actual = dense(x)
        actual_gradient, = torch.autograd.grad(actual.square().sum(), x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)
        restored = self.model()
        inject_lora(restored, rank=2, alpha=4, targets=[r'0|2'])
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)

    def test_checkpoint_with_frozen_inputs_preserves_all_adapter_gradients(self):
        model = self.model()
        inject_lora(model, rank=2, alpha=4, targets=[r'0|2'])
        for layer in model.modules():
            if isinstance(layer, LoRALinear):
                nn.init.normal_(layer.lora_B, std=.1)
        reference = copy.deepcopy(model)
        x = torch.randn(4, 5)
        expected = reference(x)
        expected.square().sum().backward()

        def run(layer, inputs):
            return (layer(inputs), None)

        actual = x
        for layer in model:
            actual, = checkpointed(run, layer, actual)
        actual.square().sum().backward()
        torch.testing.assert_close(actual, expected)
        for (name, parameter), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                torch.testing.assert_close(parameter.grad, ref.grad, msg=name)
            else:
                self.assertIsNone(parameter.grad, name)

    def test_bfloat16_frozen_base_matches_autocast_and_checkpoint(self):
        dense = nn.Sequential(nn.Linear(5, 8), nn.LayerNorm(8), nn.GELU(), nn.Linear(8, 3), nn.Linear(3, 3))
        expected, actual = copy.deepcopy(dense), copy.deepcopy(dense)
        for model, dtype in [(expected, None), (actual, 'bfloat16')]:
            torch.manual_seed(52)
            inject_lora(model, rank=2, alpha=4, targets=[r'0|3'], base_dtype=dtype)
            for layer in model.modules():
                if isinstance(layer, LoRALinear):
                    nn.init.normal_(layer.lora_B, std=.1)
        inputs = torch.randn(4, 5, requires_grad=True)
        copied = inputs.detach().clone().requires_grad_(True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            reference = expected(inputs)
            output = actual(copied)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        reference.float().square().sum().backward()
        output.float().square().sum().backward()
        torch.testing.assert_close(copied.grad, inputs.grad, rtol=0, atol=0)
        for (name, parameter), (_, ref) in zip(actual.named_parameters(), expected.named_parameters()):
            self.assertEqual(parameter.dtype, torch.bfloat16 if name in ('0.weight', '0.bias', '3.weight', '3.bias')
                             else torch.float32)
            if ref.requires_grad:
                torch.testing.assert_close(parameter.grad, ref.grad, rtol=0, atol=0, msg=name)
            else:
                self.assertIsNone(parameter.grad, name)
        restored = copy.deepcopy(dense)
        inject_lora(restored, rank=2, alpha=4, targets=[r'0|3'], base_dtype='bfloat16')
        restored.load_state_dict(actual.state_dict(), strict=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            torch.testing.assert_close(restored(copied), output, rtol=0, atol=0)

    def test_unmatched_target_does_not_freeze_or_modify_model(self):
        model = self.model()
        with self.assertRaises(ValueError):
            inject_lora(model, rank=2, alpha=4, targets=[r'0', r'no_such_layer'])
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(type(model[0]), nn.Linear)

    def test_fused_qkv_outputs_all_gradients_and_state_dict(self):
        dense = nn.Module()
        for name in ('q', 'k', 'v'):
            setattr(dense, name, nn.Linear(5, 8).double())
        separate, fused = copy.deepcopy(dense), copy.deepcopy(dense)
        for model, fuse in ((separate, False), (fused, True)):
            torch.manual_seed(55)
            inject_lora(model, rank=2, alpha=3, targets=[r'q|k|v'], fused_qkv=fuse)
            for layer in (model.q, model.k, model.v):
                nn.init.normal_(layer.lora_B, std=.1)
        self.assertTrue(fused.lora_fused_qkv)
        self.assertEqual(set(fused.state_dict()), set(separate.state_dict()))
        x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
        copied = x.detach().clone().requires_grad_(True)
        expected = tuple(layer(x) for layer in (separate.q, separate.k, separate.v))
        actual = fused_lora_qkv(fused, copied)
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=1e-12, atol=1e-12)
        sum((i + 1) * output.square().sum() for i, output in enumerate(expected)).backward()
        sum((i + 1) * output.square().sum() for i, output in enumerate(actual)).backward()
        torch.testing.assert_close(copied.grad, x.grad, rtol=1e-12, atol=1e-12)
        for (name, parameter), (_, reference) in zip(fused.named_parameters(), separate.named_parameters()):
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-12, atol=1e-12, msg=name)
            else:
                self.assertIsNone(parameter.grad, name)
        separate.load_state_dict(fused.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
