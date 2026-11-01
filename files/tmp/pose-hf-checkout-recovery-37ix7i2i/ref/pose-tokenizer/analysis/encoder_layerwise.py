"""Layer-by-layer localization of the encoder's context-starvation instability.

The shift and no-shift 32cb encoders are architecturally IDENTICAL, so we can
compare their intermediate activations layer-for-layer. For a fixed content
window (right-edge-aligned suffix), we encode it (a) WARM (with real preceding
frames) and (b) COLD (at the sequence start, zero left context), and at every
encoder submodule measure how far the COLD activation is from the WARM one at
the *first* (most-starved) content position and averaged over the window.

Tracking this deviation through depth shows WHICH layer amplifies the
starvation, and comparing the two models shows where the no-shift encoder
diverges from the (more robust) shift encoder.

Run:
    cd /home/james/pose-tokenizer
    uv run python analysis/encoder_layerwise.py
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

M = 24     # content window (frames)
P = 40     # warm prefix (frames)


@torch.no_grad()
def layerwise_starvation(m, clips):
    model = m["model"]
    acts = {}
    order = []
    handles = []

    def mk(name):
        def hook(_mod, _inp, out):
            if isinstance(out, torch.Tensor) and out.dim() == 3:
                acts[name] = out.detach()      # conv outputs are (B, C, T)
                if name not in order:
                    order.append(name)
        return hook

    # Hook only conv layers (consistent (B,C,T) layout); skip the nn.Conv1d
    # nested inside a CausalConv1d wrapper (name ends with ".conv") to avoid
    # double-counting. LayerNorm etc. use (B,T,C) and are excluded.
    for name, mod in model.encoder.named_modules():
        is_causal = mod.__class__.__name__ == "CausalConv1d"
        is_plain = isinstance(mod, torch.nn.Conv1d) and not name.endswith(".conv")
        if is_causal or is_plain:
            handles.append(mod.register_forward_hook(mk(name)))

    agg = {}  # name -> list of (dev0, rel0, devW, relW)
    n = 0
    for mi, _ in clips:
        T = mi.shape[0]
        if T < M + P:
            continue
        a = T - M
        acts.clear()
        model.encoder(model.preprocess(torch.from_numpy(mi[a:T]).float().unsqueeze(0).to(DEVICE)))
        cold = dict(acts)
        acts.clear()
        model.encoder(model.preprocess(torch.from_numpy(mi[a - P:T]).float().unsqueeze(0).to(DEVICE)))
        warm = dict(acts)
        n += 1
        for k in cold:
            if k not in warm:
                continue
            c = cold[k][0]                       # (C, Tc)  entire cold output == content window
            w = warm[k][0]                       # (C, Tw)
            nk = c.shape[-1]
            if w.shape[-1] < nk:
                continue
            wt = w[:, -nk:]                      # right-edge-aligned content window
            # first content position (most starved) and window mean
            dev0 = (c[:, 0] - wt[:, 0]).abs().mean().item()
            sc0 = wt[:, 0].abs().mean().item() + 1e-8
            devW = (c - wt).abs().mean().item()
            scW = wt.abs().mean().item() + 1e-8
            agg.setdefault(k, []).append((dev0, dev0 / sc0, devW, devW / scW))
    for h in handles:
        h.remove()
    out = {k: tuple(float(np.mean([x[i] for x in v])) for i in range(4)) for k, v in agg.items()}
    return order, out, n


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    npzs = sorted(glob.glob(NPZ_GLOB))
    res = {}
    order_ref = None
    for tag, ck in CKPTS.items():
        m = load_model(ck)
        clips = [clip_features(z, m["tf"], m["inf"], m["nc"], m["ns"]) for z in npzs]
        order, out, n = layerwise_starvation(m, clips)
        res[tag] = out
        order_ref = order
        print(f"[{tag}] layers with temporal output: {len(out)}  (n_clips used={n})")

    # Present in forward order, only temporal layers, side by side.
    print(f"\n{'layer':<34s} {'noshift rel0':>12s} {'shift rel0':>11s} {'ratio n/s':>10s} "
          f"{'noshift relW':>13s} {'shift relW':>11s}")
    rows = []
    for name in order_ref:
        if name in res["noshift"] and name in res["shift"]:
            n_rel0 = res["noshift"][name][1]; s_rel0 = res["shift"][name][1]
            n_relW = res["noshift"][name][3]; s_relW = res["shift"][name][3]
            ratio = n_rel0 / (s_rel0 + 1e-9)
            rows.append((name, n_rel0, s_rel0, ratio, n_relW, s_relW))
            print(f"{name:<34s} {n_rel0:>12.3f} {s_rel0:>11.3f} {ratio:>10.2f} {n_relW:>13.3f} {s_relW:>11.3f}")

    (OUTDIR / "encoder_layerwise.json").write_text(json.dumps(
        {"order": order_ref, "noshift": res["noshift"], "shift": res["shift"],
         "M": M, "P": P}, indent=2))
    print(f"\n[done] -> {OUTDIR/'encoder_layerwise.json'}")

    # Plot rel0 through depth.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(range(len(rows)), [r[1] for r in rows], "o-", label="noshift rel dev@first")
        ax.plot(range(len(rows)), [r[2] for r in rows], "s-", label="shift rel dev@first")
        ax.set_xticks(range(len(rows))); ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_ylabel("relative deviation (cold vs warm) @ first content pos")
        ax.set_title("Encoder context-starvation through depth (first content position)")
        ax.legend(); fig.tight_layout(); fig.savefig(OUTDIR / "encoder_layerwise.png", dpi=120); plt.close(fig)
        print(f"[plot] -> {OUTDIR/'encoder_layerwise.png'}")
    except Exception as e:
        print(f"[plot] skipped ({e})")


if __name__ == "__main__":
    main()
