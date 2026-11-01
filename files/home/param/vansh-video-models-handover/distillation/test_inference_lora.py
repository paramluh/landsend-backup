import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from distillation.lora import LoRALinear, fused_lora_qkv, inject_lora
from inference.lora import cast_inference_weights, load_lora, project_qkv, split_lora_state
from models.pose.audio import RMSNorm


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('q', 'k', 'v', 'o'):
            setattr(self, name, nn.Linear(8, 8))


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.self_attn = Attention()
        self.blocks = nn.ModuleList([block])


class Audio(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(8)
        self.q_proj = nn.Linear(8, 8)
        self.kv_proj = nn.Linear(8, 16)
        self.out_proj = nn.Linear(8, 8)

    def forward(self, x):
        x = self.norm(x)
        k, v = self.kv_proj(x).chunk(2, -1)
        return self.out_proj(F.silu(self.q_proj(x) + k) * v)


class InferenceLoRATests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(671)
        self.lora = dict(rank=2, alpha=2, base_dtype='bfloat16', fused_qkv=True,
                         targets=[r'transformer\.blocks\.\d+\.self_attn\.(?:q|k|v|o)',
                                  r'audio_adapter\.(?:q_proj|kv_proj|out_proj)'])
        self.source = nn.Module()
        self.source.transformer = Transformer().bfloat16()
        self.source.audio_adapter = Audio()
        inject_lora(self.source, **self.lora)
        with torch.no_grad():
            self.source.audio_adapter.norm.weight.fill_(1.0007)
            for name, parameter in self.source.named_parameters():
                if name.endswith('.lora_B'):
                    parameter.normal_(0, .04)
        self.source.eval().requires_grad_(False)
        self.state = {name.removeprefix('transformer.'): value.detach().clone()
                      for name, value in self.source.state_dict().items()}
        self.base, self.factors = split_lora_state(self.state, self.lora)

    def models(self):
        transformer, adapter = Transformer().bfloat16(), Audio()
        transformer.load_state_dict({name: value for name, value in self.base.items()
                                     if not name.startswith('audio_adapter.')})
        adapter.load_state_dict({name.removeprefix('audio_adapter.'): value for name, value in self.base.items()
                                 if name.startswith('audio_adapter.')})
        return transformer, adapter

    def test_load_preserves_canonical_factors_and_unmerged_function(self):
        transformer, adapter = self.models()
        names = load_lora(transformer, adapter, self.factors, self.lora)
        self.assertEqual(len(names), 7)
        self.assertEqual(transformer.blocks[0].self_attn.q.lora_A.dtype, torch.bfloat16)
        self.assertEqual(adapter.q_proj.lora_A.dtype, torch.float32)
        self.assertEqual(adapter.q_proj.weight.dtype, torch.bfloat16)
        self.assertEqual(adapter.norm.weight.dtype, torch.float32)
        self.assertTrue(all(not value.requires_grad for module in (transformer, adapter) for value in module.parameters()))
        for module, prefix in ((transformer, ''), (adapter, 'audio_adapter.')):
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value, self.state[prefix + name], rtol=0, atol=0)
        x = torch.randn(2, 3, 8)
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            expected = project_qkv(self.source.transformer.blocks[0].self_attn, x)
            actual = project_qkv(transformer.blocks[0].self_attn, x)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(adapter(x), self.source.audio_adapter(x), rtol=0, atol=0)

    def test_load_rejects_missing_extra_wrong_shape_and_silent_factor_cast(self):
        first = next(iter(self.factors))
        variants = []
        missing = dict(self.factors)
        missing.pop(first)
        variants.append(missing)
        variants.append({**self.factors, 'unused.lora_A': self.factors[first]})
        variants.append({**self.factors, first: self.factors[first][:1]})
        variants.append({**self.factors, first: self.factors[first].float()})
        for factors in variants:
            with self.subTest(keys=list(factors)[:2]):
                with self.assertRaises(ValueError):
                    load_lora(*self.models(), factors, self.lora)
        with self.assertRaises(ValueError):
            split_lora_state(self.state, None)
        with self.assertRaises(ValueError):
            split_lora_state(self.base, self.lora)
        base, factors = split_lora_state(self.base, None)
        self.assertEqual(base.keys(), self.base.keys())
        self.assertFalse(factors)

    def test_cached_weight_cast_keeps_adapter_and_norm_precision(self):
        module = Audio().eval()
        inject_lora(module, rank=2, alpha=2, targets=[r'(?:q_proj|kv_proj|out_proj)'])
        with torch.no_grad():
            module.norm.weight.fill_(1.0007)
            for name, parameter in module.named_parameters():
                if name.endswith('.lora_B'):
                    parameter.normal_(0, .04)
        module.requires_grad_(False)
        reference = copy.deepcopy(module)
        cast_inference_weights(module)
        for name, value in module.state_dict().items():
            saved = reference.state_dict()[name]
            if name.endswith(('.lora_A', '.lora_B')) or name.startswith('norm.'):
                self.assertEqual(value.dtype, torch.float32)
                torch.testing.assert_close(value, saved, rtol=0, atol=0)
            else:
                self.assertEqual(value.dtype, torch.bfloat16)
                torch.testing.assert_close(value, saved.bfloat16(), rtol=0, atol=0)
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            x = torch.randn(2, 3, 8)
            torch.testing.assert_close(module(x), reference(x), rtol=0, atol=0)
        conv = nn.Conv3d(2, 4, 1)
        weight, bias = conv.weight.detach().bfloat16(), conv.bias.detach().bfloat16()
        cast_inference_weights(conv)
        torch.testing.assert_close(conv.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(conv.bias, bias, rtol=0, atol=0)

    def test_project_qkv_dispatches_only_explicit_fused_adapters(self):
        attention = Attention()
        x = torch.randn(2, 3, 8)
        with patch('inference.lora.fused_lora_qkv', wraps=fused_lora_qkv) as fused:
            expected = (attention.q(x), attention.k(x), attention.v(x))
            for a, b in zip(project_qkv(attention, x), expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            fused.assert_not_called()
            inject_lora(attention, rank=2, alpha=2, targets=[r'(?:q|k|v)'], fused_qkv=True)
            with torch.no_grad():
                for layer in (attention.q, attention.k, attention.v):
                    self.assertIsInstance(layer, LoRALinear)
                    layer.lora_B.normal_(0, .04)
            expected = (attention.q(x), attention.k(x), attention.v(x))
            for a, b in zip(project_qkv(attention, x), expected):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
            fused.assert_called_once()


if __name__ == '__main__':
    unittest.main()
