"""Which layers does the zero->replicate padding change matter for? Encoder vs
decoder, and per single layer. Selectively enable replicate padding on subsets
of the 10 causal convs and measure the no-shift model's startup L1 (frames 0-7).
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
from pose_tokenizer.model.layers import CausalConv1d  # noqa: E402

START = slice(0, 8)
STEADY = slice(15, 40)

# per-instance controllable padding
def _fwd(self, x):
    mode = getattr(self, "_pad_mode", "constant")
    return self.conv(F.pad(x, (self.pad_left, 0), mode=mode))
layers.CausalConv1d.forward = _fwd


def label(name, part):
    n = name.replace("encoder_blocks", "b").replace("decoder_blocks", "b")
    n = n.replace(".residual_blocks.0.units.0", "").replace("_conv", "")
    return f"{part[:3]}.{n}"


def startup_steady(m, clips):
    acc = np.zeros(40); cnt = np.zeros(40)
    for mi, gt in clips:
        rec = deshift(forward_single(m["model"], mi), m["shift"])
        d = region_l1_perframe(gt, rec, 40)
        k = min(40, len(d)); acc[:k] += d[:k]; cnt[:k] += 1
    pf = acc / np.maximum(cnt, 1)
    return float(pf[START].mean()), float(pf[STEADY].mean())


def main():
    npzs = sorted(glob.glob(NPZ_GLOB))
    m = load_model(CKPTS["noshift"])
    clips = [clip_features(z, m["tf"], m["inf"], m["nc"], m["ns"]) for z in npzs]

    enc = [(label(n, "encoder"), mod) for n, mod in m["model"].encoder.named_modules() if isinstance(mod, CausalConv1d)]
    dec = [(label(n, "decoder"), mod) for n, mod in m["model"].decoder.named_modules() if isinstance(mod, CausalConv1d)]
    allc = enc + dec

    def setmodes(active):  # active = set of module ids to replicate
        for _, mod in allc:
            mod._pad_mode = "replicate" if id(mod) in active else "constant"

    def ids(pairs):
        return {id(mod) for _, mod in pairs}

    print("config                         startup   steady   startup_reduction")
    base_s, base_st = None, None
    configs = [("all zero (baseline)", set()),
               ("all replicate", ids(allc)),
               ("encoder-only", ids(enc)),
               ("decoder-only", ids(dec))]
    res = {}
    for nm, active in configs:
        setmodes(active)
        s, st = startup_steady(m, clips)
        res[nm] = s
        if nm == "all zero (baseline)":
            base_s = s
        red = "" if base_s is None else f"{(1-s/base_s)*100:+.0f}%"
        print(f"{nm:<30s} {s:>7.4f}  {st:>7.4f}   {red}")

    print("\n-- single-layer replicate (rest zero), ranked by startup reduction --")
    singles = []
    for lab, mod in allc:
        setmodes({id(mod)})
        s, _ = startup_steady(m, clips)
        singles.append((lab, s, (1 - s / base_s) * 100))
    for lab, s, red in sorted(singles, key=lambda x: x[1]):
        print(f"   {lab:<34s} startup={s:.4f}  ({red:+.0f}%)")


if __name__ == "__main__":
    main()
