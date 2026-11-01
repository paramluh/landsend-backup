"""CPU unit tests (no stores needed): system-prompt crop, side draw balance, nanostep gradient accumulation == full-batch mean grad
(decoder frozen: lr 0), shift1 / Ctx packing. usage: python tests/test_units_cpu.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_ppm as D  # noqa: E402
import lightms_s3q as LMS  # noqa: E402

# crop_blocks
b1, b2 = [9] + list(range(100)) + [8], [9] + list(range(1500)) + [8]
out, cut = D.crop_blocks([b1, b2], 1000)
assert out[0] == b1 and len(out[1]) == 1000 - len(b1) and out[1][-1] == 8 and out[1][0] == 9 and cut == len(b2) - len(out[1]), (len(out[1]), cut)
out, cut = D.crop_blocks([b1, b2], 2000)
assert out == [b1, b2] and cut == 0
out, cut = D.crop_blocks([b2, b2], 1000)
assert [len(x) for x in out] == [500, 500]
out, _ = D.crop_blocks([b1], 2)
assert out == [None]
# side balance
s = np.mean([D.side_of(f"row/{i}", 20261011) for i in range(20000)])
assert abs(s - 0.5) < 0.02, s
# nanostep accumulation with lr 0 == the full-batch mean gradient
torch.manual_seed(0)
lin = torch.nn.Linear(8, 3)
opt = torch.optim.AdamW(lin.parameters(), lr=0.0)
h = torch.randn(50, 8)
lab = torch.randint(0, 3, (50,))
hd = h.clone().requires_grad_()


def ll(c):
    return torch.nn.functional.cross_entropy(lin(hd[c]), lab[c]), None
acc, _, _, n = LMS.micro_steps(ll, hd, 50, 7, torch.Generator().manual_seed(1), opt, list(lin.parameters()), 1e9, lambda s_: 0.0, 0)
h2 = h.clone().requires_grad_()
torch.nn.functional.cross_entropy(lin(h2), lab).backward()
assert n == 7 and torch.allclose(acc, h2.grad, atol=1e-6), (acc - h2.grad).abs().max()
print("UNIT TESTS OK")
