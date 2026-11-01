"""Diagnose the source of the no-shift steady-state lag.

Test 1 (rate-distortion): decode the SAME 32cb no-shift model with the first
N=16 vs N=32 RVQ codebooks (coarser vs finer quantization, same encoder), and
compare the optimal frame-shift + the post-de-shift residual vs per-frame
acceleration. Prediction if lag is quantization smoothing: coarser (16) lags/
rounds more, and residual scales with acceleration.

Test 2 (group delay vs smoothing): build synthetic pose trajectories from two
real anchor frames -- (a) constant velocity (linear), (b) triangle oscillation
(acceleration peaks at reversals) -- encode-decode (full 32cb), and measure the
delay (constant vs speed) and peak-rounding at reversals (smoothing signature).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_lag import CKPTS, NPZ_GLOB, DEVICE, load_model, clip_features, to_coords  # noqa: E402


@torch.no_grad()
def recon_ncb(model, mi, N):
    x = torch.from_numpy(mi).float().unsqueeze(0).to(DEVICE)
    _, codes, _, _ = model.encode(x)
    rec = model.decode(codes, n_codebooks=N).squeeze(0).cpu().numpy()
    return to_coords(rec)


def shift_sweep(gt, rec, warm=8, ds=range(-3, 4)):
    T = min(len(gt), len(rec))
    g = gt[:T].reshape(T, 55, 2); r = rec[:T].reshape(T, 55, 2)
    best = {}
    for d in ds:
        if d >= 0:
            a = r[warm + d:T]; b = g[warm:T - d]
        else:
            a = r[warm:T + d]; b = g[warm - d:T]
        if len(a) < 4:
            continue
        best[d] = float(np.abs(a - b).mean())
    dstar = min(best, key=best.get)
    return dstar, best.get(0, np.nan), best[dstar]


def test1_rate(m, clips):
    print("=== Test 1: 16 vs 32 codebooks (rate-distortion), no-shift 32cb ===")
    agg = {16: [], 32: []}
    accres = {16: ([], []), 32: ([], [])}   # (acceleration, residual@+1) pooled
    for mi, gt in clips:
        for N in (16, 32):
            rec = recon_ncb(m["model"], mi, N)
            dstar, l0, lb = shift_sweep(gt, rec)
            agg[N].append((dstar, l0, lb, (l0 - lb) / l0 if l0 else np.nan))
            # post +1 de-shift residual vs acceleration, per frame
            T = min(len(gt), len(rec))
            g = gt[:T].reshape(T, 55, 2); r = rec[:T].reshape(T, 55, 2)
            acc = np.abs(g[2:] - 2 * g[1:-1] + g[:-2]).mean(axis=(1, 2))     # frames 1..T-2
            resid = np.abs(r[2:] - g[1:-1]).mean(axis=(1, 2))               # rec[t+1] vs gt[t]
            accres[N][0].extend(acc[6:].tolist()); accres[N][1].extend(resid[6:].tolist())
    for N in (16, 32):
        A = np.array(agg[N])
        med_d = np.median(A[:, 0]); drop = np.nanmean(A[:, 3])
        a = np.array(accres[N][0]); rr = np.array(accres[N][1])
        r = float(np.corrcoef(a, rr)[0, 1]); slope = float(np.polyfit(a, rr, 1)[0])
        print(f"  N={N}: median d*={med_d:+.1f}  mean L1-drop@d*={drop*100:.0f}%  "
              f"resid-vs-accel: r={r:.2f} slope={slope:.2f}  meanResid={rr.mean():.4f}")


def test2_synth(m, clips):
    print("\n=== Test 2: synthetic trajectories (group delay vs smoothing), 32cb full ===")
    # anchor frames with real motion from the highest-motion clip
    mi, gt = max(clips, key=lambda c: np.abs(np.diff(c[1], axis=0)).mean())
    T = 48
    a = mi[10].copy(); b = mi[10 + 8].copy()          # two poses ~8 frames apart
    axis = (b - a)
    # (a) constant velocity: linear ramp 0..1..(extrapolate) - use 0..1 over T
    lin = np.linspace(0, 1, T)[:, None]
    const_vel = a[None] + lin * axis[None]
    # (b) triangle oscillation: 0->1->0->1 with period P
    P = 16
    tri = 1 - np.abs(((np.arange(T) % P) / (P / 2)) - 1.0)   # 0..1..0 triangle
    triang = a[None] + tri[:, None] * axis[None]

    for name, seq, param in (("const-velocity", const_vel, lin[:, 0]),
                             ("triangle", triang, tri)):
        rec = recon_ncb(m["model"], seq.astype(np.float32), 32)
        Tn = min(len(rec), T)
        rc = rec[:Tn].reshape(Tn, 55, 2); inp = to_coords(seq[:Tn].astype(np.float32)).reshape(Tn, 55, 2)
        # delay: argmin over integer shift of |rec[t]-inp[t-d]|
        best = {}
        for d in range(-3, 4):
            if d >= 0:
                x1 = rc[8 + d:Tn]; y1 = inp[8:Tn - d]
            else:
                x1 = rc[8:Tn + d]; y1 = inp[8 - d:Tn]
            if len(x1) < 4:
                continue
            best[d] = float(np.abs(x1 - y1).mean())
        dstar = min(best, key=best.get)
        # peak-reach along the a->b axis in COORD space (for triangle: at tri==1 peaks)
        a_c = to_coords(a[None].astype(np.float32))[0].reshape(55, 2)
        b_c = to_coords(b[None].astype(np.float32))[0].reshape(55, 2)
        ax = (b_c - a_c)
        axn = (ax ** 2).sum() + 1e-8
        reach_inp = ((inp - a_c[None]) * ax[None]).sum((1, 2)) / axn
        reach_rec = ((rc - a_c[None]) * ax[None]).sum((1, 2)) / axn
        peaks = [i for i in range(1, Tn - 1) if param[i] > 0.98] if name == "triangle" else []
        msg = f"  {name}: delay d*={dstar:+d}  err@d*={best[dstar]:.4f}  err@0={best.get(0,float('nan')):.4f}"
        if peaks:
            # recon reach near each peak (max in +-2 window, to be fair to delay)
            pr = np.mean([max(reach_rec[max(0, p - 2):p + 3]) for p in peaks])
            pi = np.mean([reach_inp[p] for p in peaks])
            msg += f"  | peak reach input={pi:.2f} recon={pr:.2f} (undershoot={1-pr/pi:+.0%})"
        print(msg)


def main():
    npzs = sorted(glob.glob(NPZ_GLOB))
    m = load_model(CKPTS["noshift"])
    clips = [clip_features(z, m["tf"], m["inf"], m["nc"], m["ns"]) for z in npzs]
    test1_rate(m, clips)
    test2_synth(m, clips)


if __name__ == "__main__":
    main()
