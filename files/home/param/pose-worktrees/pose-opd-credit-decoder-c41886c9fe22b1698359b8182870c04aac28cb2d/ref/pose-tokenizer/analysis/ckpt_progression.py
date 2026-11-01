"""Track the startup/lag/encoder-robustness metrics across training checkpoints
for the shift and no-shift models at matched steps (10k, 50k, 150k).

Answers: do the shift-vs-no-shift behaviours (encoder cold-start robustness gap,
startup ramp, +1 lag) emerge early in training or only late?

Run:
    cd /home/james/pose-tokenizer
    uv run python analysis/ckpt_progression.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_lag import (  # noqa: E402
    NPZ_GLOB, OUTDIR, load_model, clip_features,
    A1_encoder_prefix, B_shift_sweep, region_l1_perframe, forward_single,
)

NB = "/mnt/somfs/james-checkpoints/pose-tokenizer/base-causal-pretrain-32cb-pose20k-subset1.2tb"
SB = "/mnt/somfs/james-checkpoints/pose-tokenizer-james/32cb_size_2048/base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048"
STEPS = [10000, 50000, 150000]


def metrics_for(ck, npzs):
    m = load_model(ck)
    clips = [clip_features(z, m["tf"], m["inf"], m["nc"], m["ns"]) for z in npzs]
    named = [(Path(z).name.replace("_rtmwx.npz", ""), c) for z, c in zip(npzs, clips)]

    # encoder context-dependence at zero prefix (A1): code disagreement + latent L1
    a1 = A1_encoder_prefix(m["model"], clips)
    enc_code0 = a1[0][0]
    enc_lat0 = a1[0][1]

    # raw startup ramp (no de-shift, no discard): peak(first 8)/steady(15-25)
    acc = np.zeros(30); cnt = np.zeros(30)
    for mi, gt in clips:
        rec = forward_single(m["model"], mi)
        d = region_l1_perframe(gt, rec, 30)
        n = min(30, len(d)); acc[:n] += d[:n]; cnt[:n] += 1
    ramp = acc / np.maximum(cnt, 1)
    steady = float(ramp[15:25].mean())
    ramp_ratio = float(ramp[:8].max() / (steady + 1e-9))

    # lag (B): median optimal shift + mean L1 drop at d* (region all)
    rows = B_shift_sweep(m["model"], named)
    dstars = [r["regions"]["all"]["dstar"] for r in rows]
    drops = [r["regions"]["all"]["drop_frac"] for r in rows]
    return dict(shift_cfg=m["shift"], enc_code0=enc_code0, enc_lat0=enc_lat0,
                ramp_ratio=ramp_ratio, steady=steady,
                lag_dstar_med=float(np.median(dstars)),
                lag_drop_mean=float(np.mean(drops)))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    npzs = sorted(glob.glob(NPZ_GLOB))
    out = {}
    for step in STEPS:
        for tag, base in (("noshift", NB), ("shift", SB)):
            ck = f"{base}/step_{step}"
            print(f"[run] {tag} step {step} ...", flush=True)
            out[f"{tag}_{step}"] = metrics_for(ck, npzs)

    print(f"\n{'ckpt':<16s} {'cfgshift':>8s} {'encLatDev0':>10s} {'encCodeDev0':>11s} "
          f"{'rawRamp':>8s} {'steady':>8s} {'lag d*':>7s} {'lagDrop':>8s}")
    for step in STEPS:
        for tag in ("noshift", "shift"):
            r = out[f"{tag}_{step}"]
            print(f"{tag+' @'+str(step//1000)+'k':<16s} {r['shift_cfg']:>8d} {r['enc_lat0']:>10.4f} "
                  f"{r['enc_code0']:>11.3f} {r['ramp_ratio']:>8.1f} {r['steady']:>8.4f} "
                  f"{r['lag_dstar_med']:>7.1f} {r['lag_drop_mean']*100:>7.0f}%")

    # ratios (noshift/shift) per step for the encoder-robustness gap
    print("\nencoder latent-dev0 ratio (noshift/shift) by step:")
    for step in STEPS:
        n = out[f"noshift_{step}"]["enc_lat0"]; s = out[f"shift_{step}"]["enc_lat0"]
        print(f"   step {step//1000}k: {n/max(s,1e-9):.2f}x  (noshift {n:.4f} vs shift {s:.4f})")

    (OUTDIR / "ckpt_progression.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {OUTDIR/'ckpt_progression.json'}")


if __name__ == "__main__":
    main()
