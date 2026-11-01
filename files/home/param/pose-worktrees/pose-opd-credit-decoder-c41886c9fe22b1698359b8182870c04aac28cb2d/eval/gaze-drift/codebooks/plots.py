import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "/data/home/amartyavarma/ppm/gaze-drift/codebooks"
R = json.load(open(f"{OUT}/results.json"))
COL = {"GT": "#2a78d6", "prefill": "#1baf7a", "fs": "#eb6834", "d1": "#eda100", "d3": "#e87ba4", "head": "#2a78d6",
       "body": "#eb6834", "hands": "#1baf7a"}
INK, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "lines.linewidth": 2})
cb = np.arange(32)
what = sys.argv[1:] or ["sens", "chg", "fix"]

if "sens" in what and "sens" in R:
    S = R["sens"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, (mode, ttl) in enumerate([("random", "codebook c -> random codes"), ("hold", "codebook c -> previous frame's code")]):
        for g, lab in [("head_sw", "head (nose/eyes/ears)"), ("body", "arms/hips"), ("hands", "hands")]:
            ax[i].plot(cb, S[mode][g], "-o", ms=3, color=COL[g.split("_")[0]], label=lab)
        ax[i].set_yscale("log")
        ax[i].set_xlabel("codebook index (RVQ depth)")
        ax[i].set_ylabel("mean |change| (shoulder widths)")
        ax[i].set_title(ttl)
        ax[i].legend(frameon=False)
    ax[2].plot(cb + 1, S["trunc"]["head_sw"], "-o", ms=3, color=COL["head"], label="head")
    ax[2].plot(cb + 1, S["trunc"]["body"], "-o", ms=3, color=COL["body"], label="arms/hips")
    ax[2].plot(cb + 1, S["trunc"]["hands"], "-o", ms=3, color=COL["hands"], label="hands")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("codebooks kept (0..k-1)")
    ax[2].set_ylabel("error vs all 32 (shoulder widths)")
    ax[2].set_title("truncation")
    ax[2].legend(frameon=False)
    fig.suptitle("Head sensitivity per pose codebook (GT tokens, 11 clips)", color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cb_head_sensitivity.png", dpi=110)

if "chg" in what and "stats" in R:
    S = R["stats"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, lab in [("GT", "GT (2-12 s)"), ("prefill", "real prefill (0-2 s)"), ("fs", "gen fs: no CFG, light k50"),
                   ("d1", "gen d1: CFG 1, light k30"), ("d3", "gen d3: CFG 3, light k30")]:
        ax[0].plot(cb, S[k]["chg"], "-o", ms=3, color=COL[k], label=lab)
        ax[1].plot(cb, S[k]["H"], "-o", ms=3, color=COL[k], label=lab)
    ax[0].set_xlabel("codebook index")
    ax[0].set_ylabel("code-change rate per token frame")
    ax[0].set_title("code-change rate (gen 2-12 s)")
    ax[1].set_xlabel("codebook index")
    ax[1].set_ylabel("unigram entropy (bits)")
    ax[1].set_title("unigram entropy (equal N; prefill N is 5x smaller)")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cb_change_rate.png", dpi=110)

if "fix" in what:
    res = {}
    for k in ("fixes", "fixes_0", "fixes_1"):
        if k in R:
            res.update(R[k]["res"])
    order = [f for f in ["none", "trunc24", "trunc16", "trunc8", "trunc4", "gtmode16", "gtmode8", "hold2_16", "hold2_8", "hold2_1",
                         "mode3_8", "mode3_1", "latsm1_8", "latsm1_1", "latsm2_1", "kpsm1", "kpsm2"] if f in res]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    x = np.arange(len(order))
    for j, (met, ttl) in enumerate([("head_jit", "head jitter (HF RMS, ear-dist units)"), ("yaw_lf", "slow yaw drift (0.5 s low-pass std)")]):
        g = res["none"]["GT"]["matched"][met]
        for i, (s, off) in enumerate([("GT", -0.3), ("fs", -0.1), ("d1", 0.1), ("d3", 0.3)]):
            v = [res[f][s]["matched"][met] for f in order]
            ax[j].bar(x + off, v, 0.18, color=COL[s], label={"GT": "GT tokens + same fix", "fs": "gen fs", "d1": "gen d1",
                                                              "d3": "gen d3"}[s])
        ax[j].axhline(g, color=INK, lw=1, ls="--", label="GT, no fix")
        ax[j].set_xticks(x, order, rotation=60, ha="right")
        ax[j].set_title(ttl + ", 2-12 s")
        ax[j].set_ylabel(met)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{OUT}/cb_fix_comparison.png", dpi=110)
print("plots ok", what)
