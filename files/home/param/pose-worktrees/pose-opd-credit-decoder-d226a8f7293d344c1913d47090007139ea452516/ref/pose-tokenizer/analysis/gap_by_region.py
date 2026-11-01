"""How much does replicate padding close the no-shift vs shift GAP, split into
startup frames (0-7) vs steady state (15+)?

For each model x pad-mode, compute per-frame L1 (native de-shift applied, i.e.
deployed form), then the startup-region and steady-region means, and the
no-shift-minus-shift gap in each region under zero vs replicate padding.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_tokenizer.model.layers as layers  # noqa: E402
from startup_lag import (  # noqa: E402
    CKPTS, NPZ_GLOB, load_model, clip_features,
    forward_single, region_l1_perframe, deshift,
)

START = slice(0, 8)
STEADY = slice(15, 40)


def set_pad(mode):
    def fwd(self, x):
        return self.conv(F.pad(x, (self.pad_left, 0), mode=mode))
    layers.CausalConv1d.forward = fwd


def perframe(m, clips, nf=40):
    acc = np.zeros(nf); cnt = np.zeros(nf)
    for mi, gt in clips:
        rec = deshift(forward_single(m["model"], mi), m["shift"])   # deployed form
        d = region_l1_perframe(gt, rec, nf)
        n = min(nf, len(d)); acc[:n] += d[:n]; cnt[:n] += 1
    return acc / np.maximum(cnt, 1)


def main():
    npzs = sorted(glob.glob(NPZ_GLOB))
    models = {}
    for tag, ck in CKPTS.items():
        m = load_model(ck)
        m["clips"] = [clip_features(z, m["tf"], m["inf"], m["nc"], m["ns"]) for z in npzs]
        models[tag] = m

    res = {}
    for mode in ("constant", "replicate"):
        set_pad(mode)
        for tag in ("noshift", "shift"):
            pf = perframe(models[tag], models[tag]["clips"])
            res[(mode, tag)] = (float(pf[START].mean()), float(pf[STEADY].mean()))
    set_pad("constant")

    lbl = {"constant": "zero", "replicate": "replicate"}
    print(f"{'pad':<10s} {'region':<8s} {'noshift':>9s} {'shift':>9s} {'gap(ns-sh)':>11s} {'gap ratio':>9s}")
    for mode in ("constant", "replicate"):
        for ri, region in enumerate(("startup", "steady")):
            ns = res[(mode, "noshift")][ri]; sh = res[(mode, "shift")][ri]
            print(f"{lbl[mode]:<10s} {region:<8s} {ns:>9.4f} {sh:>9.4f} {ns-sh:>+11.4f} {ns/sh:>8.2f}x")

    print("\n=== how much replicate closes the gap ===")
    for ri, region in enumerate(("startup", "steady")):
        gz = res[("constant", "noshift")][ri] - res[("constant", "shift")][ri]
        gr = res[("replicate", "noshift")][ri] - res[("replicate", "shift")][ri]
        closed = (1 - gr / gz) * 100 if abs(gz) > 1e-9 else float("nan")
        print(f"{region:<8s}: gap {gz:+.4f} (zero) -> {gr:+.4f} (replicate)  => {closed:.0f}% of the gap closed")


if __name__ == "__main__":
    main()
