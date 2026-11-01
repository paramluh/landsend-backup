import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

G = "/data/home/amartyavarma/ppm/gaze-drift/geom"
R = json.load(open(f"{G}/results.json"))
EX = R["extras"]["summary"]
z = np.load(f"{G}/kp.npz")
CLIP = {"S-05": "10_27286bdd", "S-13": "06_74807fe0", "S-15": "10_27286bdd", "S-20": "19_47473a46", "E-03": "03_b437950e",
        "E-05": "05_e25c6859", "E-09": "09_d9ef3c94", "E-12": "12_72cb45d3", "E-13": "13_956c09af", "E-15": "15_d1e1322c",
        "E-18": "18_1ffa9083"}
ROWS = list(CLIP)
COL = {"GT": "#2a78d6", "PF": "#8a8984", "fs": "#eb6834", "d1": "#1baf7a", "d2": "#eda100", "d3": "#e87ba4"}
INK, MUT = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUT, "axes.labelcolor": INK, "xtick.color": MUT, "ytick.color": MUT,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2, "figure.facecolor": "white"})


def yaw(kp):
    xy = kp[..., :2]
    ear_mid = (xy[:, 3] + xy[:, 4]) / 2
    ear_d = np.linalg.norm(xy[:, 3] - xy[:, 4], axis=-1)
    return (xy[:, 0, 0] - ear_mid[:, 0]) / ear_d


# (a) yaw proxy vs time, small multiples per row: GT (12 s) vs fs vs d3
fig, axs = plt.subplots(3, 4, figsize=(11, 6.2), sharex=True)
for ax, r in zip(axs.flat, ROWS):
    for c, key in [("GT", f"gtpkg__{CLIP[r]}"), ("fs", f"fs__{r}"), ("d3", f"d3__{r}")]:
        y = yaw(z[key])
        ax.plot(np.arange(len(y)) / 25, y, color=COL[c], lw=1.0 if c != "GT" else 1.4, label={"GT": "GT (12 s)", "fs": "fs (no CFG)", "d3": "d3 (CFG 3)"}[c])
    ax.axvspan(0, 2, color="#d9d8d3", alpha=0.6, lw=0)
    ax.set_title(f"{r}  ({CLIP[r]})", fontsize=8, color=INK)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
ax = axs.flat[-1]
ax.axis("off")
h, l = axs.flat[0].get_legend_handles_labels()
ax.legend(h + [plt.Rectangle((0, 0), 1, 1, color="#d9d8d3")], l + ["real prefill 0-2 s"], loc="center", frameon=False, fontsize=9)
for ax in axs[-1]:
    ax.set_xlabel("time (s)")
fig.supylabel("yaw proxy = (nose x - ear-mid x) / ear distance", fontsize=9)
fig.suptitle("Head yaw proxy vs time: GT vs generated (fs, d3). Generated is flatter, not wandering", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(f"{G}/a_yaw_vs_time.png", dpi=110)
plt.close(fig)

# (b) deviation from prefill mean vs time, median + IQR, 4 panels (yaw_ear, pitch, roll, absolute nose x)
cur = R["curves"]
fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, (key, lab) in zip(axs, [("yaw_ear", "|yaw - prefill mean|"), ("pitch", "|pitch - prefill mean|"), ("roll", "|roll - prefill mean| (deg)"),
                                ("absnose_x", "|nose x - prefill mean| (shoulder widths)")]):
    for c in ["GT", "fs", "d1", "d2", "d3"]:
        d = cur[key][c]
        t = np.array(d["t"])
        m = np.array(d["med"], float)
        ok = np.isfinite(m)
        ax.plot(t[ok], m[ok], color=COL[c], label=c, lw=1.8 if c == "GT" else 1.2)
        ax.fill_between(t[ok], np.array(d["q25"], float)[ok], np.array(d["q75"], float)[ok], color=COL[c], alpha=0.12, lw=0)
    ax.axvspan(0, 2, color="#d9d8d3", alpha=0.6, lw=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(lab)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
axs[0].legend(frameon=False, fontsize=7, ncol=2)
fig.suptitle("Deviation from the sample's own 2 s prefill mean (median over 11 rows, IQR band; GT ends at 12 s)", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(f"{G}/b_dev_from_prefill.png", dpi=110)
plt.close(fig)

# (c) head jitter + low-frequency motion bars per condition
S = R["summary"]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.0))
conds = ["PF", "GT", "fs", "d1", "d2", "d3"]
for ax, (k, lab) in zip(axs, [("H1_jit", "head jitter |Δ|/frame (eye dists), rel. shoulders"), ("still_frac_010", "fraction of frames head ~still (<0.01 eye-dist/frame)"),
                              ("H4_conf_head", "mean head-keypoint confidence")]):
    vals = []
    for c in conds:
        bn = "0-2" if c == "PF" else "2-12"
        vals.append((EX if k == "still_frac_010" else S)[k][bn].get("GT" if c == "PF" and k == "still_frac_010" else c))
    x = np.arange(len(conds))
    ax.bar(x, vals, color=[COL[c] for c in conds], width=0.7)
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, ["prefill\n0-2 s", "GT\n2-12 s", "fs", "d1", "d2", "d3"])
    ax.set_ylabel(lab)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
fig.suptitle("Matched 2-12 s window, median over rows: generated head moves ~2x LESS than GT and is still 3-5x more often; confidences equal", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(f"{G}/c_jitter_bars.png", dpi=110)
plt.close(fig)

# extra numbers: static offset of the generated head vs GT (|window mean - prefill mean| / GT std), 2-12 s
P = R["per_sample"]
off = {}
for key in ["yaw_ear", "pitch", "roll", "absnose_x", "headpos_x"]:
    off[key] = {}
    for c in ["GT", "fs", "d1", "d2", "d3"]:
        v = [abs(P[f"{c}/{r}/2-12"][f"{key}_mean"] - P[f"{c}/{r}/0-2"][f"{key}_mean"]) for r in ROWS]
        sd = [P[f"{c}/{r}/2-12"][f"{key}_std"] for r in ROWS]
        off[key][c] = {"offset_med": float(np.median(v)), "std_med": float(np.median(sd)), "offset_over_std_med": float(np.median(np.array(v) / np.array(sd)))}
R["static_offset_2-12"] = off
json.dump(R, open(f"{G}/results.json", "w"), indent=1)
for k, v in off.items():
    print(k, {c: (round(o["offset_med"], 4), round(o["std_med"], 4), round(o["offset_over_std_med"], 2)) for c, o in v.items()})
print("plots ok")
