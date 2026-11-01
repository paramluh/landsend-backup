"""D31 check (CPU): the 0.6B decoder update loop covers EVERY light frame exactly once per micro, in K disjoint chunks.

Runs the REAL train_ppm.nanosteps source (extracted with ast, exec'd with stubs: a recording decoder, a no-op all-reduce, a stub optimizer;
the only textual edit is dropping .pin_memory(), which needs CUDA). Also checks the frame pickers at frac 1.0 return every candidate frame:
pose_s3q.light_pick (pose decoder + speech decoder on pose rows) and speech_q.light_select (speech decoder on speech rows)."""
import ast
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "lib"))
src = open(os.path.join(CODE, "train_ppm.py")).read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "nanosteps")
code = ast.get_source_segment(src, fn).replace(".pin_memory()", "")


class Dist:
    @staticmethod
    def all_reduce(t):
        return t


class Opt:
    def __init__(self):
        self.param_groups = [{"lr": 0.0}]
        self.steps = 0

    def step(self):
        self.steps += 1


class Do:
    def __init__(self):
        self.p = torch.nn.Parameter(torch.zeros(4))
        self.ps, self.rep, self.flat, self.opt, self.micros = [self.p], True, torch.zeros(4), Opt(), 7

    def lr(self, s):
        return 1e-4


def run(n, K, MB):
    rec = []                                                       # per update j: the valid frames of each micro-batch, and the weights

    def dec(hd, depth, fl, wf, n_valid=0):
        assert fl.numel() == MB and wf.numel() == MB
        assert (wf[n_valid:] == 0).all()
        rec[-1].append((fl[:n_valid].tolist(), wf[:n_valid].tolist()))
        l_ = (hd.float().sum() * 0.0) + (wf * 0.0).sum()
        return l_, torch.zeros(()), n_valid, torch.zeros(31)

    class Rec:                                                     # one list per update: hook the optimizer step boundary
        pass
    g = {"torch": torch, "CFG": {"dec_k": K, "dec_micro_frames": MB, "dec_seed": 20261011, "dec_max_frames": 7500, "dec_clip": 0.5},
         "RANK": 3, "WORLD": 128, "dist": Dist}
    exec(code, g)
    do = Do()
    orig = do.opt.step

    def step():
        orig(); rec.append([])
    do.opt.step = step
    rec.append([])
    light = torch.from_numpy(np.sort(np.random.default_rng(n).choice(40000, n, replace=False))).long()
    hv = torch.randn(40000, 8)
    acc, st, lrs = g["nanosteps"](dec, do, hv, None, light)
    rec.pop()                                                      # the empty list opened after the last step
    assert len(rec) == K == do.opt.steps == len(lrs), (len(rec), K, do.opt.steps)
    chunks = [sum((f for f, _ in r), []) for r in rec]
    allf = sum(chunks, [])
    assert len(allf) == n and sorted(allf) == light.tolist(), "frames used != all light frames exactly once"
    for a in range(K):
        for b in range(a + 1, K):
            assert not set(chunks[a]) & set(chunks[b]), "chunks overlap"
        m = len(chunks[a])
        w = sum(sum(x for x in ws) for _, ws in rec[a])
        assert m == 0 or abs(w - 1.0 / 31.0) < 1e-6, ("chunk weights do not form the chunk mean over frames x 31", w)
    sz = [len(c) for c in chunks]
    assert max(sz) - min(sz) <= 1, sz
    return sz


for n in (1, 7, 37, 299, 300, 2401, 2400, 7499, 7500, 7501, 9000, 24000):
    sz = run(n, 8, 300)
    print(f"n={n:6d} K=8 MB=300: 8 disjoint chunks of sizes {sz}, union == all {n} frames, each exactly once")
for n in (7500, 7501):
    print(f"n={n} K=4 (old) sizes {run(n, 4, 300)}")

import pose_s3q as PS  # noqa: E402
import speech_q as SP  # noqa: E402
rng = np.random.default_rng(0)
for t in range(200):
    pos = np.sort(rng.choice(100000, int(rng.integers(1, 9000)), replace=False))
    assert np.array_equal(PS.light_pick(pos, 1.0, (20261001, t, 1, 0x9053)), pos)
    d = rng.integers(-1, 5, size=(int(rng.integers(2, 5000)), 32))
    v = np.nonzero((d[:, 0] >= 0) & (d[:, 1] >= 0))[0]
    if len(v):
        assert np.array_equal(SP.light_select(d, 1.0, (20261001, 1, 2, t, 0, 1)), v)
print("frac 1.0: pose_s3q.light_pick and speech_q.light_select return EVERY candidate frame (200 random cases each)")
print("D31 PARTITION TEST PASSED")
