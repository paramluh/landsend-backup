"""H7 analysis of gaze_series.json -> results.json (+ curves / scatter PNGs). CPU, small."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

P = "/data/home/amartyavarma/ppm/gaze-drift/path"
GEOM = "/data/home/amartyavarma/ppm/gaze-drift/geom/results.json"
FPS = 25
UNSEEN = {"19_47473a46", "15_d1e1322c", "18_1ffa9083"}
ROWS = dict([("S-05", "10_27286bdd"), ("S-13", "06_74807fe0"), ("S-15", "10_27286bdd"), ("S-20", "19_47473a46"), ("E-03", "03_b437950e"),
             ("E-05", "05_e25c6859"), ("E-09", "09_d9ef3c94"), ("E-12", "12_72cb45d3"), ("E-13", "13_956c09af"), ("E-15", "15_d1e1322c"),
             ("E-18", "18_1ffa9083")])
COND = ["GT", "fs", "d1", "d2", "d3"]
COL = {"GT": "#0b0b0b", "fs": "#2a78d6", "d1": "#eb6834", "d2": "#1baf7a", "d3": "#eda100"}
LAB = {"GT": "GT pose render", "fs": "gen no-CFG k50", "d1": "gen CFG1", "d2": "gen CFG2", "d3": "gen CFG3"}
D = json.load(open(f"{P}/gaze_series.json"))
S, PIX = D["series"], D["pixdiff_vs_gt"]
# metrics: deviation from the 0-2 s mean (|x - mean_0-2|); ncc: 1 - ncc (already relative to the 0-0.4 s face)
MET = {"iris_x": "iris offset x (gaze L/R), |dev| / eye-window width", "iris_y": "iris offset y (gaze up/down), |dev| / eye-window width",
       "sym": "face symmetry axis (yaw proxy), |dev| / IOD", "dx": "face x position, |dev| / IOD", "dy": "face y position, |dev| / IOD",
       "lscale": "face scale, |log(scale / 0-2 s)|", "appear": "1 - NCC(face, 0-0.4 s face)"}


def dev(rec, m):
    if m == "appear":
        return 1 - np.asarray(rec["ncc"], float)
    x = np.asarray(rec["scale"] if m == "lscale" else rec[m], float)
    if m == "lscale":
        x = np.log(x)
    if m in ("iris_x", "iris_y", "sym"):   # light smoothing (0.2 s median-ish) to tame per-frame centroid noise
        k = 5
        xp = np.pad(x, (k // 2, k // 2), mode="edge")
        x = np.array([np.nanmedian(xp[i:i + k]) for i in range(len(x))])
    return np.abs(x - np.nanmean(x[:2 * FPS]))


def key_series():
    """-> {(cond, row): rec}; GT rows map to the clip's GT render."""
    out = {}
    for row, idn in ROWS.items():
        out[("GT", row)] = S[f"GT/{idn}"]
        for c in COND[1:]:
            out[(c, row)] = S[f"{c}/{row}"]
    return out


KS = key_series()
WIN = {"0-2": (0, 2), "2-6": (2, 6), "6-12": (6, 12), "2-12": (2, 11.8), "12-20": (12, 19.9)}
res = {"note": "deviation from each video's own 0-2 s mean, measured in the rendered video (no face model on disk: pose-anchored template "
               "tracking + dark-centroid iris + mirror-symmetry yaw proxy). GT renders end at 11.88 s. unseen = video-model unseen-eval clips "
               "(19_47473a46 S-20, 15_d1e1322c E-15, 18_1ffa9083 E-18); train = the 7 rows whose video chunk is in the video model's train split.",
       "windows": {}, "per_row": {}}
for m in MET:
    res["windows"][m] = {}
    for grp, rows in (("all", list(ROWS)), ("unseen", [r for r in ROWS if ROWS[r] in UNSEEN]), ("train", [r for r in ROWS if ROWS[r] not in UNSEEN])):
        res["windows"][m][grp] = {}
        for c in COND:
            res["windows"][m][grp][c] = {}
            for w, (a, b) in WIN.items():
                vals = []
                for r in rows:
                    d = dev(KS[(c, r)], m)
                    seg = d[int(a * FPS):int(b * FPS)]
                    if len(seg) > FPS:
                        vals.append(float(np.nanmean(seg)))
                if vals:
                    res["windows"][m][grp][c][w] = round(float(np.mean(vals)), 4)
# per-row 2-12 gen vs GT, + geom stillness
G = json.load(open(GEOM))["extras"]["per_sample"]
for c in COND[1:]:
    for r in ROWS:
        o = {m: round(float(np.nanmean(dev(KS[(c, r)], m)[2 * FPS:int(11.8 * FPS)])), 4) for m in MET}
        o.update({f"GT_{m}": round(float(np.nanmean(dev(KS[("GT", r)], m)[2 * FPS:int(11.8 * FPS)])), 4) for m in MET})
        o.update({f"{m}_12-20": round(float(np.nanmean(dev(KS[(c, r)], m)[12 * FPS:])), 4) for m in MET})
        g = G.get(f"{c}/{r}/2-12", {})
        o["still_frac_010"] = g.get("still_frac_010")
        o["head_spd"] = g.get("head_spd")
        o["unseen"] = ROWS[r] in UNSEEN
        res["per_row"][f"{c}/{r}"] = o
# correlations across the 44 gen samples
corr = {}
pr = res["per_row"]
for m in ("iris_x", "iris_y", "sym", "appear", "lscale"):
    x = [pr[k]["still_frac_010"] for k in pr if pr[k]["still_frac_010"] is not None]
    y = [pr[k][m] - pr[k][f"GT_{m}"] for k in pr if pr[k]["still_frac_010"] is not None]
    y2 = [pr[k][m] for k in pr if pr[k]["still_frac_010"] is not None]
    rho, p = spearmanr(x, y)
    rho2, p2 = spearmanr(x, y2)
    corr[m] = {"spearman_still_vs_(gen-GT)_2-12": [round(float(rho), 3), round(float(p), 4)], "spearman_still_vs_gen_2-12": [round(float(rho2), 3), round(float(p2), 4)], "n": len(x)}
res["corr_with_geom_stillness"] = corr
# fraction of rows where gen 2-12 dev > GT 2-12 dev
frac = {}
for m in MET:
    frac[m] = {c: round(float(np.mean([pr[f"{c}/{r}"][m] > pr[f"{c}/{r}"][f"GT_{m}"] for r in ROWS])), 3) for c in COND[1:]}
res["frac_rows_gen_gt_GT_2-12"] = frac
# 12-20 vs 2-12 for generated (rollout-length effect)
late = {}
for m in MET:
    late[m] = {c: [res["windows"][m]["all"][c].get("2-12"), res["windows"][m]["all"][c].get("12-20")] for c in COND[1:]}
res["gen_2-12_vs_12-20"] = late
# pixel diff GT vs gen render per second (same clip): identical inputs in 0-2 s
pdv = {}
for k, v in PIX.items():
    v = np.asarray(v)
    pdv[k] = {"0-1.9": round(float(v[:47].mean()), 3), "2.2-4": round(float(v[55:100].mean()), 3), "4-11.8": round(float(v[100:].mean()), 3)}
res["pixdiff_gt_vs_gen_mean_abs_8bit"] = {"per_sample": pdv,
                                          "mean": {w: round(float(np.mean([d[w] for d in pdv.values()])), 3) for w in ("0-1.9", "2.2-4", "4-11.8")}}
# video-level head motion (face-centre speed, IOD per frame, 2-11.8 s) + iris motion: is the rendered head as still as the pose?
vm = {}
for c in COND:
    sp, ir = [], []
    for r in ROWS:
        rec = KS[(c, r)]
        x, y = np.asarray(rec["dx"], float), np.asarray(rec["dy"], float)
        s = slice(2 * FPS, int(11.8 * FPS))
        sp.append(float(np.nanmean(np.hypot(np.diff(x[s]), np.diff(y[s])))))
        ix = np.asarray(rec["iris_x"], float)[s]
        ir.append(float(np.nanstd(ix)))
    vm[c] = {"face_speed_iod_per_frame": round(float(np.mean(sp)), 4), "iris_x_std": round(float(np.mean(ir)), 4)}
res["video_motion_2-12"] = vm
json.dump(res, open(f"{P}/results.json", "w"), indent=1)


# ---- plot 1: curves (1 s bins) for gaze/yaw/appearance, unseen vs train
def binned(c, rows, m):
    arr = []
    for r in rows:
        d = dev(KS[(c, r)], m)
        nb = len(d) // FPS
        arr.append(np.array([np.nanmean(d[i * FPS:(i + 1) * FPS]) for i in range(nb)] + [np.nan] * (20 - nb))[:20])
    return np.nanmean(np.stack(arr), 0)


mets = ["iris_x", "sym", "appear"]
fig, axs = plt.subplots(len(mets), 2, figsize=(10, 8), sharex=True)
for j, (grp, rows) in enumerate((("video-model UNSEEN clips (3 rows)", [r for r in ROWS if ROWS[r] in UNSEEN]),
                                 ("video-model TRAIN clips (8 rows)", [r for r in ROWS if ROWS[r] not in UNSEEN]))):
    for i, m in enumerate(mets):
        ax = axs[i, j]
        ax.axvspan(0, 2, color="#d9d9d9", alpha=0.6, lw=0)
        ax.axvline(12, color="#999999", lw=1, ls=":")
        t = np.arange(20) + 0.5
        for c in COND:
            y = binned(c, rows, m)
            ax.plot(t, y, color=COL[c], lw=2.4 if c == "GT" else 1.6, label=LAB[c], marker="o" if c == "GT" else None, ms=3)
        ax.set_title(f"{grp}" if i == 0 else "", fontsize=9)
        ax.set_ylabel(MET[m].split(",")[0], fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
axs[-1, 0].set_xlabel("time in video (s); grey = 2 s real-pose prefill; dotted = end of GT render (11.9 s)", fontsize=8)
axs[-1, 1].set_xlabel("time in video (s)", fontsize=8)
axs[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
fig.suptitle("Rendered-video drift from the 0-2 s mean vs time (mean over rows, 1 s bins)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{P}/b_video_drift_vs_time.png", dpi=90)

# ---- plot 2: per-row gaze drift (gen - GT, 2-12 s) vs geom head stillness
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for c in COND[1:]:
    ks = [f"{c}/{r}" for r in ROWS]
    x = [pr[k]["still_frac_010"] for k in ks]
    ax[0].scatter(x, [pr[k]["iris_x"] - pr[k]["GT_iris_x"] for k in ks], color=COL[c], s=22, label=LAB[c], edgecolor="white", lw=0.6)
    ax[1].scatter(x, [pr[k]["appear"] - pr[k]["GT_appear"] for k in ks], color=COL[c], s=22, label=LAB[c], edgecolor="white", lw=0.6)
for a, m in zip(ax, ("iris_x", "appear")):
    a.axhline(0, color="#999999", lw=1)
    a.set_xlabel("head still fraction 2-12 s (geom, decoded pose)", fontsize=8)
    a.set_ylabel(f"gen - GT, 2-12 s: {MET[m].split(',')[0]}", fontsize=8)
    a.set_title(f"Spearman rho {corr[m]['spearman_still_vs_(gen-GT)_2-12'][0]} (p {corr[m]['spearman_still_vs_(gen-GT)_2-12'][1]})", fontsize=8)
    a.grid(alpha=0.25, lw=0.5)
    a.tick_params(labelsize=7)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
ax[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(f"{P}/c_drift_vs_stillness.png", dpi=90)
print(json.dumps({k: res[k] for k in ("gen_2-12_vs_12-20", "frac_rows_gen_gt_GT_2-12", "corr_with_geom_stillness")}, indent=0)[:4000])
for m in MET:
    print(m, json.dumps(res["windows"][m]))
print("pixdiff", res["pixdiff_gt_vs_gen_mean_abs_8bit"]["mean"])
print('video_motion', res['video_motion_2-12'])
