"""Find real aloe 2s subclips whose dominant motion is most constant-acceleration
(quadratic) and test whether no-shift flattens curvature while shift preserves it
(as the synthetic const-accel probe suggested, but now in-distribution).

Window motion is projected onto its top PCA direction -> 1D signal s(t). A window
is "constant-acceleration-like" when a quadratic fits it very well AND clearly
beats a linear fit (real curvature, low jerk). We feed the WHOLE clip through the
model (so the mid-clip window is warm, no cold-start), extract the window's recon,
project onto the same axis, and compare quadratic curvature and delay.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_lag import CKPTS, NPZ_GLOB, DEVICE, load_model, clip_features, to_coords, deshift  # noqa: E402

W = 50   # 2 s at 25 fps


@torch.no_grad()
def enc_dec(model, mi):
    x = torch.from_numpy(mi.astype(np.float32)).unsqueeze(0).to(DEVICE)
    return to_coords(model(x)["reconstruction"].squeeze(0).cpu().numpy())


def r2(y, yhat):
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - yhat) ** 2).sum() / (ss + 1e-12)


def quad_lin_fit(t, s):
    cq = np.polyfit(t, s, 2); cl = np.polyfit(t, s, 1)
    return cq, r2(s, np.polyval(cq, t)), r2(s, np.polyval(cl, t))


def pc1_axis(win_coords):   # (W,110) -> unit axis (110,)
    X = win_coords - win_coords.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[0]


def main():
    npzs = sorted(glob.glob(NPZ_GLOB))
    mN = load_model(CKPTS["noshift"])
    clips = [(Path(z).name.replace("_rtmwx.npz", ""), clip_features(z, mN["tf"], mN["inf"], mN["nc"], mN["ns"]))
             for z in npzs]

    # --- rank windows by constant-acceleration-ness ---
    cands = []
    for name, (mi, gt) in clips:
        Tn = len(gt)
        for w in range(0, Tn - W, 5):
            win = gt[w:w + W]
            ax = pc1_axis(win)
            s = (win - win.mean(0)) @ ax
            t = np.arange(W)
            cq, R2q, R2l = quad_lin_fit(t, s)
            var = s.var()
            if R2q > 0.8 and (R2q - R2l) > 0.1 and var > 1e-4:
                cands.append((R2q - R2l, R2q, R2l, name, w, ax, var, abs(cq[0])))
    cands.sort(reverse=True)
    print(f"scanned; {len(cands)} candidate windows with R2q>0.8 and clear curvature\n")
    print("top constant-acceleration-like 2s windows:")
    for sc, R2q, R2l, name, w, ax, var, acc in cands[:5]:
        print(f"  {name} [{w}:{w+W}]  R2_quad={R2q:.3f} R2_lin={R2l:.3f} (quad-lin={sc:.3f})  |accel|~{acc:.2e}")

    models = {"noshift": (mN, 0), "shift": (load_model(CKPTS["shift"]), 1)}

    print("\n=== curvature-tracking on the top real windows (fed in full-clip context) ===")
    ratios = {"noshift": [], "shift": []}
    for sc, R2q, R2l, name, w, ax, var, acc in cands[:6]:
        mi, gt = dict((n, c) for n, c in clips)[name]
        t = np.arange(W)
        s_in = (gt[w:w + W] - gt[w:w + W].mean(0)) @ ax
        cq_in = np.polyfit(t, s_in, 2)
        print(f"\n {name} [{w}:{w+W}]  (R2_quad={R2q:.3f})")
        for tag, (m, ns) in models.items():
            rec = deshift(enc_dec(m["model"], mi), ns)
            Tn = min(len(rec), len(gt))
            if w + W > Tn:
                continue
            s_rec = (rec[w:w + W] - gt[w:w + W].mean(0)) @ ax
            cq_rec = np.polyfit(t, s_rec, 2)
            ratio = float(cq_rec[0] / cq_in[0]) if abs(cq_in[0]) > 1e-9 else float("nan")
            # fractional delay: minimize |s_rec(t) - s_in(t - tau)| using quadratic model of s_in
            taus = np.arange(-3, 3.01, 0.1)
            errs = [np.abs(s_rec[4:W - 4] - np.polyval(cq_in, (t - tau)[4:W - 4])).mean() for tau in taus]
            tau = float(taus[int(np.argmin(errs))])
            fit_r2 = r2(s_rec, np.polyval(cq_rec, t))
            if np.isfinite(ratio):
                ratios[tag].append(ratio)
            print(f"    {tag:<8s} curvature_ratio={ratio:+.2f}  delay tau*={tau:+.2f}  recon_quadR2={fit_r2:.2f}")
    print("\n=== mean curvature ratio over windows ===")
    for tag in ("noshift", "shift"):
        if ratios[tag]:
            print(f"  {tag}: mean={np.mean(ratios[tag]):.2f}  (n={len(ratios[tag])}, vals={[round(x,2) for x in ratios[tag]]})")


if __name__ == "__main__":
    main()
