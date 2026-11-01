"""Test the 'irreducible causal group delay' claim on constant-acceleration motion.

Build a synthetic trajectory where each joint moves along a fixed axis with a
parametric position s(t): linear (constant velocity) or quadratic (constant
acceleration). Encode-decode, project the reconstruction back onto the axis to
get s_rec(t), and measure:
  - delay tau = argmin_tau || s_rec(t) - s_input(t - tau) ||   (fractional)
  - curvature ratio = accel(s_rec) / accel(s_input)
A pure group delay: de-shift -> ~0 residual, curvature ratio ~1 (shifted parabola).
Smoothing: curvature ratio < 1 (parabola flattened).
Compare no-shift (raw) vs shift (native de-shift +1): if the delay is group delay,
no-shift shows tau~+1 with curvature~1, and shift shows tau~0.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_lag import CKPTS, NPZ_GLOB, DEVICE, load_model, clip_features, to_coords, deshift  # noqa: E402

T = 48
WARM = 10          # ignore the first WARM frames (cold-start) in the fit
TAIL = 4


@torch.no_grad()
def enc_dec(model, seq):
    x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(DEVICE)
    return to_coords(model(x)["reconstruction"].squeeze(0).cpu().numpy())   # (T,110)


def s_of(kind, u):
    uu = np.clip(u, 0, T - 1) / (T - 1)
    return uu ** 2 if kind == "accel" else uu


def analyze(model, seq, a_c, b_c, kind, native_shift):
    rec = enc_dec(model, seq)
    rec = deshift(rec, native_shift)              # deployed form
    ax = (b_c - a_c); axn = (ax ** 2).sum() + 1e-8
    s_rec = ((rec.reshape(-1, 55, 2) - a_c[None]) * ax[None]).sum((1, 2)) / axn
    n = min(len(s_rec), T)
    tgrid = np.arange(n)
    # delay: minimize over fractional tau
    taus = np.arange(-3, 3.01, 0.1)
    errs = []
    for tau in taus:
        pred = s_of(kind, tgrid - tau)
        e = np.abs(s_rec[WARM:n - TAIL] - pred[WARM:n - TAIL]).mean()
        errs.append(e)
    j = int(np.argmin(errs)); tau_star = float(taus[j]); err_star = float(errs[j])
    err0 = float(errs[np.argmin(np.abs(taus))])
    # curvature ratio (second difference), interior
    acc_rec = np.diff(s_rec[WARM:n - TAIL], 2).mean()
    acc_in = np.diff(s_of(kind, tgrid[WARM:n - TAIL]), 2).mean()
    ratio = float(acc_rec / acc_in) if abs(acc_in) > 1e-9 else float("nan")
    return tau_star, err_star, err0, ratio


def main():
    npzs = sorted(glob.glob(NPZ_GLOB))
    # anchors from a high-motion clip
    mN = load_model(CKPTS["noshift"])
    clips = [clip_features(z, mN["tf"], mN["inf"], mN["nc"], mN["ns"]) for z in npzs]
    mi, gt = max(clips, key=lambda c: np.abs(np.diff(c[1], axis=0)).mean())
    a = mi[10].copy(); b = mi[18].copy()
    a_c = to_coords(a[None].astype(np.float32))[0].reshape(55, 2)
    b_c = to_coords(b[None].astype(np.float32))[0].reshape(55, 2)

    models = {"noshift": (mN, 0), "shift": (load_model(CKPTS["shift"]), 1)}
    for kind in ("velocity", "accel"):
        s = s_of(kind, np.arange(T))
        seq = a[None] + s[:, None] * (b - a)[None]
        print(f"\n=== constant-{kind} trajectory ===")
        for tag, (m, ns) in models.items():
            tau, e_star, e0, ratio = analyze(m["model"], seq, a_c, b_c, kind, ns)
            extra = f"  curvature_ratio={ratio:.2f}" if kind == "accel" else ""
            print(f"  {tag:<8s} (deshift {ns}):  delay tau*={tau:+.2f}  "
                  f"err@tau*={e_star:.4f}  err@0={e0:.4f}{extra}")


if __name__ == "__main__":
    main()
