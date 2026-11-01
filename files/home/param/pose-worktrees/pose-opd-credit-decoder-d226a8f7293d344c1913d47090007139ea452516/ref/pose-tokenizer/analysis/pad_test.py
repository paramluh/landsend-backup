"""Prediction #2 (inference-only): does swapping causal zero-pad -> replicate
(edge) padding reduce the cold-start, and does it help no-shift MORE than shift?

Weights were trained with zero-pad, so this is an out-of-distribution
intervention (suggestive, not conclusive). We monkeypatch CausalConv1d.forward
to change ONLY the pad mode, then compare for both models:
  - encoder context-deviation at zero prefix (A1 latent L1)
  - raw startup ramp (peak(first8)/steady)
  - steady-state L1 (guard: replicate must not wreck the bulk)

Run:
    cd /home/james/pose-tokenizer
    uv run python analysis/pad_test.py
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
from startup_lag import (  # noqa: E402
    CKPTS, NPZ_GLOB, load_model, clip_features,
    A1_encoder_prefix, forward_single, region_l1_perframe,
)


def set_pad_mode(mode):
    def fwd(self, x):
        return self.conv(F.pad(x, (self.pad_left, 0), mode=mode))
    layers.CausalConv1d.forward = fwd


def ramp_and_steady(m, clips):
    acc = np.zeros(30); cnt = np.zeros(30)
    for mi, gt in clips:
        rec = forward_single(m["model"], mi)
        d = region_l1_perframe(gt, rec, 30)
        n = min(30, len(d)); acc[:n] += d[:n]; cnt[:n] += 1
    r = acc / np.maximum(cnt, 1)
    steady = float(r[15:25].mean())
    return float(r[:8].max() / (steady + 1e-9)), steady, float(r[0])


def main():
    npzs = sorted(glob.glob(NPZ_GLOB))
    print(f"{'model':<9s} {'pad':<10s} {'encLatDev0':>11s} {'rawRamp':>8s} {'frame0':>8s} {'steady':>8s}")
    results = {}
    for tag, ck in CKPTS.items():
        m = load_model(ck)
        clips = [clip_features(z, m["tf"], m["inf"], m["nc"], m["ns"]) for z in npzs]
        for mode in ("constant", "replicate"):   # constant==zero-pad (baseline)
            set_pad_mode(mode)
            latdev0 = A1_encoder_prefix(m["model"], clips)[0][1]
            ramp, steady, f0 = ramp_and_steady(m, clips)
            results[(tag, mode)] = dict(latdev0=latdev0, ramp=ramp, steady=steady, f0=f0)
            lbl = "zero" if mode == "constant" else "replicate"
            print(f"{tag:<9s} {lbl:<10s} {latdev0:>11.4f} {ramp:>8.1f} {f0:>8.4f} {steady:>8.4f}")
    # restore
    set_pad_mode("constant")

    print("\n=== effect of replicate (relative to zero-pad) ===")
    for tag in ("noshift", "shift"):
        z = results[(tag, "constant")]; r = results[(tag, "replicate")]
        print(f"{tag}: frame0 {z['f0']:.4f} -> {r['f0']:.4f} ({(1-r['f0']/z['f0'])*100:+.0f}%)   "
              f"encLatDev0 {z['latdev0']:.4f} -> {r['latdev0']:.4f} ({(1-r['latdev0']/z['latdev0'])*100:+.0f}%)   "
              f"steady {z['steady']:.4f} -> {r['steady']:.4f} ({(r['steady']/z['steady']-1)*100:+.0f}%)")


if __name__ == "__main__":
    main()
