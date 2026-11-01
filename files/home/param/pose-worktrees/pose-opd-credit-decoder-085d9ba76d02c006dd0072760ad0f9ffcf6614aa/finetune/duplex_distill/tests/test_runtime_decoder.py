"""Execute the actual decoder_kl body on tiny CPU decoders, without GPU imports."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from finetune.duplex_distill.losses import distribution_kl


def load_decoder_kl():
    path = Path(__file__).resolve().parents[1] / 'runtime.py'
    module = ast.parse(path.read_text(), filename=str(path))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'decoder_kl')
    code = compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec')
    namespace = {'torch': torch, 'distribution_kl': distribution_kl,
                 'profile': SimpleNamespace(span=lambda name: nullcontext())}
    exec(code, namespace)
    return namespace['decoder_kl']


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Parameter(torch.randn(31, 7, 3, dtype=torch.float64) * .1)
        self.bias = nn.Parameter(torch.randn(31, 7, dtype=torch.float64) * .1)

    def forward(self, hv, depth, flat, zb=None):
        hidden = hv.index_select(0, flat)
        if zb is not None:
            hidden = hidden + zb.index_select(0, flat)
        return torch.einsum('nh,dvh->ndv', hidden, self.head) + self.bias, depth.index_select(0, flat)[:, 1:]


class RuntimeDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(95)

    def test_scale_applied_once_to_decoder_parameters_tower_and_link_roots(self):
        decoder_kl = load_decoder_kl()
        for scale in (.75 / 3, 1.):
            with self.subTest(objective_scale=scale):
                student, teacher = TinyDecoder(), TinyDecoder()
                reference = TinyDecoder(); reference.load_state_dict(student.state_dict())
                base = torch.randn(39, 3, dtype=torch.float64, requires_grad=True)
                link_base = torch.randn(39, 3, dtype=torch.float64, requires_grad=True)
                ref_base = base.detach().clone().requires_grad_()
                ref_link_base = link_base.detach().clone().requires_grad_()
                teacher_hv = torch.randn(39, 3, dtype=torch.float64, requires_grad=True)
                teacher_zb = torch.randn(39, 3, dtype=torch.float64, requires_grad=True)
                hv, zb = base * 1.5, link_base * 2.5
                depth = torch.randint(0, 7, (39, 32))
                selected = torch.arange(2, 37)  # 35 frames => 16+16+3 chunks.
                world, global_frames = 6, 70
                dh, dz, metric = decoder_kl(student, teacher, hv, teacher_hv, depth, selected,
                    zb=zb, teacher_zb=teacher_zb, global_frames=global_frames, world=world,
                    dummy=False, chunk_cap=17, step=9, rank=2, seed=123, objective_scale=scale)
                torch.autograd.backward([hv, zb], [dh, dz])
                with torch.no_grad():
                    target, _ = teacher(teacher_hv, depth, selected, zb=teacher_zb)
                logits, _ = reference(ref_base * 1.5, depth, selected, zb=ref_link_base * 2.5)
                teacher_logp = target.log_softmax(-1)
                per_code = (teacher_logp.exp() * (teacher_logp - logits.log_softmax(-1))).sum(-1)
                # Same fp32 normalized frame weight as the runtime, with a
                # wholly independent full-distribution reference objective.
                normalizer = float(torch.tensor(world / (31 * global_frames), dtype=torch.float32))
                expected = per_code.sum() * normalizer * scale
                expected.backward()
                torch.testing.assert_close(base.grad, ref_base.grad, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(link_base.grad, ref_link_base.grad, atol=1e-12, rtol=1e-12)
                for actual, wanted in zip(student.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, wanted.grad, atol=1e-12, rtol=1e-12)
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                self.assertIsNone(teacher_hv.grad); self.assertIsNone(teacher_zb.grad)
                torch.testing.assert_close(metric['kl'], per_code.detach().sum(0), atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(metric['count'], torch.full((31,), 35., dtype=torch.float64))

    def test_dataset_ce_diagnostics_have_no_gradient_contribution(self):
        decoder_kl = load_decoder_kl()
        student, teacher = TinyDecoder(), TinyDecoder()
        hidden = torch.randn(21, 3, dtype=torch.float64, requires_grad=True)
        teacher_hidden = torch.randn(21, 3, dtype=torch.float64)
        selected = torch.arange(21)
        gradients, ce_values = [], []
        for label in (0, 6):
            student.zero_grad(set_to_none=True)
            dh, dz, stats = decoder_kl(student, teacher, hidden, teacher_hidden,
                torch.full((21, 32), label), selected, zb=None, teacher_zb=None,
                global_frames=21, world=1, dummy=False, chunk_cap=16, step=1, rank=0, seed=88,
                objective_scale=.75 / 3)
            gradients.append([dh.clone(), *[p.grad.clone() for p in student.parameters()]])
            ce_values.append(float(stats['ce'].sum()))
            self.assertIsNone(dz)
            self.assertTrue(all(not value.requires_grad for value in stats.values()))
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertNotEqual(*ce_values)


if __name__ == '__main__':
    unittest.main()
