"""Real Qwen layer math/gradient equivalence for per-layer checkpointing on CPU."""
import ast
import copy
from pathlib import Path
import types
import unittest

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config
from transformers.models.qwen3 import modeling_qwen3 as Q3


HERE = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    module = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def cpu_attn32(q, k, v, scale):
    # Production kernel layout is [batch,time,heads,head_dim]. CPU SDPA is an
    # attention-kernel substitute only; all actual Qwen RMSNorm/projection/RoPE/
    # MLP modules and the production/new trunk function bodies remain unchanged.
    repeat = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(repeat, dim=2)
    v = v.repeat_interleave(repeat, dim=2)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                        is_causal=True, dropout_p=0., scale=scale)
    return out.transpose(1, 2)


class DecoderCheckpointTests(unittest.TestCase):
    def test_actual_qwen_forward_and_all_gradients_match_production(self):
        torch.set_num_threads(1)
        da = types.SimpleNamespace(attn32=cpu_attn32)
        old = load_functions(HERE / 'production' / 'model_ppm.py', {'_dec_trunk'},
                             {'torch': torch, 'Q3': Q3, 'DA': da})['_dec_trunk']
        new = load_functions(HERE / 'runtime.py', {'decoder_layer', 'decoder_trunk_checkpointed'},
                             {'torch': torch, 'M': types.SimpleNamespace(Q3=Q3, DA=da, checkpoint=checkpoint)})['decoder_trunk_checkpointed']
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                torch.manual_seed(930)
                cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=8, attention_dropout=0., rms_norm_eps=1e-6)
                model = Q3.Qwen3Model(cfg).to(dtype)
                recomputed = copy.deepcopy(model)
                x = torch.randn(3, 32, 32, dtype=dtype, requires_grad=True)
                xr = x.detach().clone().requires_grad_()
                reference = old(model, x)
                actual = new(recomputed, xr)
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                upstream = torch.randn_like(actual)
                (reference * upstream).sum().backward()
                (actual * upstream).sum().backward()
                torch.testing.assert_close(xr.grad, x.grad, atol=0, rtol=0)
                ref_params = dict(model.named_parameters())
                checked = 0
                for name, param in recomputed.named_parameters():
                    expected = ref_params[name].grad
                    if expected is None:
                        self.assertIsNone(param.grad, name)
                    else:
                        torch.testing.assert_close(param.grad, expected, atol=0, rtol=0, msg=name)
                        checked += 1
                self.assertGreater(checked, 30)


if __name__ == '__main__':
    unittest.main()
