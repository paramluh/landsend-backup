"""Two mechanistic tests of the "no-shift = predictive/derivative coder" hypothesis.

Test 1 (residual-magnitude law): the frame-0 encoder-latent deviation (cold vs
warm) should scale with the boundary residual ||x0 - pad||. Sweep pad in {zero,
replicate} over many (clip, crop) boundaries and regress deviation on residual.
Prediction: strong ~linear-through-origin law for no-shift, much weaker for shift.

Test 2 (derivative structure): perturb two adjacent input frames (t, t-1) in
common mode (both +d) vs differential mode (+d, -d) and measure a deep token's
response. A derivative/predictive code responds to the DIFFERENCE (common mode
cancels). Prediction: no-shift more differential-dominated than shift.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_tokenizer.model.layers as layers  # noqa: E402
from startup_lag import CKPTS, NPZ_GLOB, DEVICE, load_model, clip_features  # noqa: E402

_PAD = {"mode": "constant"}
def _fwd(self, x):
    return self.conv(F.pad(x, (self.pad_left, 0), mode=_PAD["mode"]))
layers.CausalConv1d.forward = _fwd


@torch.no_grad()
def enc_latent(model, arr):
    x = torch.from_numpy(arr).float().unsqueeze(0).to(DEVICE)
    return model.encoder(model.preprocess(x)).squeeze(0)   # (C, T')


def test1_residual_law(m, clips, M=24, P=40, crops_per_clip=6):
    model = m["model"]
    rng = np.random.default_rng(0)
    xs, ys = [], []          # residual, deviation  (pooled over pad modes)
    dz, dr = [], []          # deviations split by pad
    for mi, _ in clips:
        T = mi.shape[0]
        if T < M + P + 4:
            continue
        starts = rng.integers(P, T - M, size=min(crops_per_clip, T - M - P))
        for a in starts:
            x0 = mi[a]                              # boundary frame (input space)
            res_zero = float(np.linalg.norm(x0))
            _PAD["mode"] = "constant"
            zw = enc_latent(model, mi[a - P:a + M])   # warm reference (prefix present)
            first_warm = zw[:, -(M // 2)]             # token of first content frame
            zc = enc_latent(model, mi[a:a + M])       # cold, zero pad
            dev_zero = float((zc[:, 0] - first_warm).abs().mean())
            _PAD["mode"] = "replicate"
            zr = enc_latent(model, mi[a:a + M])       # cold, replicate pad
            dev_rep = float((zr[:, 0] - first_warm).abs().mean())
            xs += [res_zero, 0.0]; ys += [dev_zero, dev_rep]
            dz.append(dev_zero); dr.append(dev_rep)
    _PAD["mode"] = "constant"
    xs = np.array(xs); ys = np.array(ys)
    # line through origin: k = argmin ||y - k x||
    k = float((xs * ys).sum() / (xs * xs).sum())
    resid = ys - k * xs
    r2 = 1 - (resid ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    r = float(np.corrcoef(xs, ys)[0, 1])
    return dict(slope=k, r2=float(r2), r=r, dev_zero=float(np.mean(dz)),
                dev_rep=float(np.mean(dr)), n=len(dz))


def test2_derivative(m, clips, delta=0.05, n_dir=8):
    model = m["model"]
    rng = np.random.default_rng(1)
    _PAD["mode"] = "constant"
    ratios = []
    for mi, _ in clips:
        T = mi.shape[0]
        if T < 40:
            continue
        z0 = enc_latent(model, mi)
        Ttok = z0.shape[1]
        for tau in [Ttok // 2, (2 * Ttok) // 3]:
            j = 2 * tau                                 # newest input frame for token tau
            if j < 2 or j >= T:
                continue
            for _ in range(n_dir):
                u = rng.standard_normal(mi.shape[1]).astype(np.float32)
                u = u / (np.linalg.norm(u) + 1e-8) * delta
                xc = mi.copy(); xc[j] += u; xc[j - 1] += u          # common mode
                xd = mi.copy(); xd[j] += u; xd[j - 1] -= u          # differential
                zc = enc_latent(model, xc); zd = enc_latent(model, xd)
                dcom = float((zc[:, tau] - z0[:, tau]).abs().sum())
                ddif = float((zd[:, tau] - z0[:, tau]).abs().sum())
                if dcom + ddif > 1e-9:
                    ratios.append(ddif / (dcom + ddif))
    return dict(derivativeness=float(np.mean(ratios)), n=len(ratios))


def main():
    npzs = sorted(glob.glob(NPZ_GLOB))
    for tag, ck in CKPTS.items():
        m = load_model(ck)
        clips = [clip_features(z, m["tf"], m["inf"], m["nc"], m["ns"]) for z in npzs]
        t1 = test1_residual_law(m, clips)
        t2 = test2_derivative(m, clips)
        print(f"\n=== {tag} ===")
        print(f"  [Test1 residual law] dev(zero)={t1['dev_zero']:.4f}  dev(replicate)={t1['dev_rep']:.4f}  "
              f"n={t1['n']}")
        print(f"     deviation ~= k*||x0-pad||  (line through origin): k={t1['slope']:.4f}  "
              f"R^2={t1['r2']:.3f}  pearson r={t1['r']:.3f}")
        print(f"  [Test2 derivative]  differential-share = {t2['derivativeness']:.3f}  "
              f"(0.5=balanced, ->1 = pure derivative/difference coder)  n={t2['n']}")


if __name__ == "__main__":
    main()
