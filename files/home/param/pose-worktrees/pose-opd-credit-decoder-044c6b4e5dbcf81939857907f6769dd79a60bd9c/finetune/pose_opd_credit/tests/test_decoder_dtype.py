"""Decoder replay preserves the cached actor's FP32 residual arithmetic."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from finetune.pose_opd_credit import runtime


class Norm(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.71, 1.23, hidden))
        self.variance_epsilon = 1e-6

    def forward(self, value):
        dtype = value.dtype
        value = value.float()
        return self.weight * (value * torch.rsqrt(value.square().mean(-1, keepdim=True)
                                                   + self.variance_epsilon)).to(dtype)


def rotary(q, k, cos, sin, unsqueeze_dim):
    def rotate(value):
        left, right = value.chunk(2, -1)
        return torch.cat((-right, left), -1)
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


def attention(q, k, v, scale):
    # This reference has equal Q/K head counts; production uses GQA.
    result = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), is_causal=True, scale=scale)
    return result.transpose(1, 2)


class DepthDtypeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)

    def test_slot0_addition_and_embeddings_match_actor_fp32_upcasts(self):
        decoder = nn.Module()
        decoder.proj = nn.Linear(3, 4).bfloat16()
        decoder.cb0_emb = nn.Embedding(2048, 4).bfloat16()
        decoder.model = nn.Module()
        decoder.model.embed_tokens = nn.Embedding(31 * 2048, 4).bfloat16()
        hv = torch.randn(4, 3).bfloat16()
        zb = torch.randn(4, 4).bfloat16()
        depth = torch.randint(0, 2048, (4, 32)) + torch.arange(32) * 2048
        selected = torch.tensor([3, 1])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            x, labels = runtime._decoder_inputs(decoder, hv, depth, selected, zb)
            projection = decoder.proj(hv[selected])
        self.assertEqual(x.dtype, torch.float32)
        torch.testing.assert_close(x[:, 0], projection.float() + zb[selected].float(), rtol=0, atol=0)
        self.assertFalse(torch.equal(x[:, 0], (projection + zb[selected]).float()))
        torch.testing.assert_close(x[:, 1], decoder.cb0_emb(depth[selected, 0]).float(), rtol=0, atol=0)
        torch.testing.assert_close(labels, depth[selected, 1:] - torch.arange(1, 32) * 2048)
        x.square().sum().backward()
        self.assertIsNotNone(decoder.proj.weight.grad)
        self.assertIsNotNone(decoder.cb0_emb.weight.grad)

    def test_layer_matches_fp32_upcast_actor_parameters_under_bf16_autocast(self):
        layer = nn.Module()
        layer.input_layernorm, layer.post_attention_layernorm = Norm(8), Norm(8)
        layer.self_attn = nn.Module()
        a = layer.self_attn
        a.head_dim, a.scaling = 4, .5
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(a, name, nn.Linear(8, 8, bias=False))
        a.q_norm, a.k_norm = Norm(4), Norm(4)
        layer.mlp = nn.Sequential(nn.Linear(8, 16, bias=False), nn.SiLU(), nn.Linear(16, 8, bias=False))
        layer.bfloat16()
        # These are the actor's exact rounded values, upcast without recovering
        # information from the learner's original FP32 master parameters.
        reference = copy.deepcopy(layer).float()
        h = torch.randn(2, 5, 8, requires_grad=True)
        angles = torch.randn(1, 5, 4)
        cos, sin = angles.cos(), angles.sin()
        native = SimpleNamespace(Q3=SimpleNamespace(apply_rotary_pos_emb=rotary),
                                 DA=SimpleNamespace(attn32=attention))
        with patch.object(runtime, 'M', native), torch.autocast('cpu', dtype=torch.bfloat16):
            actual = runtime._depth_layer(layer, h, cos, sin)
            # Independent ordinary HF-style FP32 module forward. Q/K norms
            # explicitly retain the normalized BF16 projection before multiply.
            ra = reference.self_attn
            y = reference.input_layernorm(h)
            q, k = ra.q_norm(ra.q_proj(y).view(2, 5, 2, 4)), ra.k_norm(ra.k_proj(y).view(2, 5, 2, 4))
            v = ra.v_proj(y).view(2, 5, 2, 4)
            q, k = rotary(q, k, cos, sin, 2)
            z = attention(q, k, v.float(), .5)
            residual = h + ra.o_proj(z.reshape(2, 5, 8))
            expected = residual + reference.mlp(reference.post_attention_layernorm(residual))
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertTrue(torch.isfinite(a.q_norm.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
