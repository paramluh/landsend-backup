"""Prediction #1: encoder temporal-memory profile (no training needed).

For a deep encoder output token, backprop to the input and measure how much the
token depends on the input frame at lag k (frames before the token's newest
input). Hypothesis: the no-shift encoder has LONGER memory (heavier weight on
large lag / deep past) than the shift encoder, which would explain why it is
starved specifically at the clip start (where deep past is absent).

Run:
    cd /home/james/pose-tokenizer
    uv run python analysis/temporal_memory.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_lag import CKPTS, NPZ_GLOB, OUTDIR, DEVICE, load_model, clip_features  # noqa: E402

MAXLAG = 20


def memory_profile(m, clips):
    model = m["model"]
    prof = np.zeros(MAXLAG + 1)
    cnt = 0
    for mi, _ in clips:
        x = torch.from_numpy(mi).float().unsqueeze(0).to(DEVICE)
        x.requires_grad_(True)
        z = model.encoder(model.preprocess(x))          # (1, C, T')
        Ttok = z.shape[-1]
        # probe a few deep interior tokens (full past, away from right edge)
        for t0 in [Ttok // 2, (2 * Ttok) // 3, Ttok - 3]:
            if t0 < 2 or t0 >= Ttok:
                continue
            if x.grad is not None:
                x.grad = None
            (z[0, :, t0] ** 2).sum().backward(retain_graph=True)
            g = x.grad[0].abs().sum(1).detach().cpu().numpy()   # (T,) per input frame
            nz = np.nonzero(g > g.max() * 1e-4)[0]
            if len(nz) == 0:
                continue
            R = int(nz.max())                              # newest input frame the token sees
            for lag in range(MAXLAG + 1):
                j = R - lag
                if 0 <= j < len(g):
                    prof[lag] += g[j]
            cnt += 1
    prof = prof / max(cnt, 1)
    prof = prof / (prof.sum() + 1e-12)                     # normalize to a distribution over lag
    return prof


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    npzs = sorted(glob.glob(NPZ_GLOB))
    res = {}
    for tag, ck in CKPTS.items():
        m = load_model(ck)
        clips = [clip_features(z, m["tf"], m["inf"], m["nc"], m["ns"]) for z in npzs]
        res[tag] = memory_profile(m, clips)

    print(f"{'lag(frames back)':<18s} {'noshift':>10s} {'shift':>10s}")
    for lag in range(MAXLAG + 1):
        print(f"{lag:<18d} {res['noshift'][lag]:>10.4f} {res['shift'][lag]:>10.4f}")

    def eff_mem(p):
        lags = np.arange(len(p))
        mean_lag = float((lags * p).sum())
        cum = np.cumsum(p)
        lag90 = int(np.searchsorted(cum, 0.90))
        return mean_lag, lag90

    for tag in ("noshift", "shift"):
        mm, l90 = eff_mem(res[tag])
        print(f"\n{tag}: weighted-mean lag = {mm:.2f} frames, 90%-cumulative lag = {l90} frames")

    (OUTDIR / "temporal_memory.json").write_text(json.dumps(
        {t: res[t].tolist() for t in res}, indent=2))
    print(f"\n[done] -> {OUTDIR/'temporal_memory.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(MAXLAG + 1), res["noshift"], "o-", label="noshift")
        ax.plot(range(MAXLAG + 1), res["shift"], "s-", label="shift")
        ax.set_xlabel("lag (input frames before token's newest input)")
        ax.set_ylabel("normalized gradient weight")
        ax.set_title("Encoder temporal-memory profile")
        ax.legend(); fig.tight_layout(); fig.savefig(OUTDIR / "temporal_memory.png", dpi=120); plt.close(fig)
        print(f"[plot] -> {OUTDIR/'temporal_memory.png'}")
    except Exception as e:
        print(f"[plot] skipped ({e})")


if __name__ == "__main__":
    main()
