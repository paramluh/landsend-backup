"""Late-drift feature scan: S-05 and E-03 fs (plus all 11 fs rows) per-second time courses vs the clip's GT 2-12 s 5-95% band.
-> late_S05_E03.png, late_drift.json"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

G = "/data/home/amartyavarma/ppm/gaze-drift/geom"
FPS, PF = 25, 50
CLIP = {"S-05": "10_27286bdd", "S-13": "06_74807fe0", "S-15": "10_27286bdd", "S-20": "19_47473a46", "E-03": "03_b437950e",
        "E-05": "05_e25c6859", "E-09": "09_d9ef3c94", "E-12": "12_72cb45d3", "E-13": "13_956c09af", "E-15": "15_d1e1322c",
        "E-18": "18_1ffa9083"}
ROWS = list(CLIP)
z = np.load(f"{G}/kp.npz")
nrm = lambda v: np.linalg.norm(v, axis=-1)  # noqa: E731


def feats(kp, ref):
    """per-frame features; normalisers from `ref` (the GT clip's own prefill, identical to the sample's prefill)."""
    xy, cf = kp[..., :2], kp[..., 2]
    sw0 = np.median(nrm(ref[:PF, 5, :2] - ref[:PF, 6, :2]))
    sh = (xy[:, 5] + xy[:, 6]) / 2
    eye_mid, ear_mid = (xy[:, 1] + xy[:, 2]) / 2, (xy[:, 3] + xy[:, 4]) / 2
    eye_d, ear_d = nrm(xy[:, 1] - xy[:, 2]), nrm(xy[:, 3] - xy[:, 4])
    nose = xy[:, 0]
    F = {}
    F["pitch (nose-eye_mid y)/eye_d"] = (nose[:, 1] - eye_mid[:, 1]) / eye_d
    F["yaw (nose-ear_mid x)/ear_d"] = (nose[:, 0] - ear_mid[:, 0]) / ear_d
    F["head x rel shoulder mid (sw0)"] = (nose[:, 0] - sh[:, 0]) / sw0
    F["head y rel shoulder mid (sw0)"] = (nose[:, 1] - sh[:, 1]) / sw0
    F["eye_d / ear_d"] = eye_d / ear_d
    F["shoulder width / prefill"] = nrm(xy[:, 5] - xy[:, 6]) / sw0
    F["shoulder mid x (canvas px)"] = sh[:, 0]
    F["shoulder mid y (canvas px)"] = sh[:, 1]
    d = xy[:, 5] - xy[:, 6]
    F["shoulder tilt (deg)"] = np.degrees(np.arctan2(d[:, 1], np.abs(d[:, 0])))
    for side, e, w in [("L", 7, 9), ("R", 8, 10)]:
        F[f"elbow {side} height above sh mid (sw0)"] = (sh[:, 1] - xy[:, e, 1]) / sw0
        F[f"wrist {side} height above sh mid (sw0)"] = (sh[:, 1] - xy[:, w, 1]) / sw0
    for side, sl, wr in [("L", slice(13, 34), 9), ("R", slice(34, 55), 10)]:
        pts = np.concatenate([xy[:, sl], xy[:, wr:wr + 1]], 1)
        c = np.concatenate([cf[:, sl], cf[:, wr:wr + 1]], 1)
        dist = nrm(pts - nose[:, None]) / sw0
        dist = np.where(c > 0.3, dist, np.inf)
        m = dist.min(1)
        F[f"hand {side} to nose min dist (sw0, conf>0.3)"] = np.where(np.isfinite(m), m, np.nan)
        F[f"hand {side} mean conf"] = c.mean(1)
    F["head conf mean"] = cf[:, :5].mean(1)
    return F


def per_sec(x):
    n = len(x) // FPS
    return np.array([np.nanmean(x[i * FPS:(i + 1) * FPS]) if np.isfinite(x[i * FPS:(i + 1) * FPS]).any() else np.nan for i in range(n)])


GTF, FSF, BAND = {}, {}, {}
for r in ROWS:
    gt = z[f"gtpkg__{CLIP[r]}"]
    GTF[r] = feats(gt, gt)
    FSF[r] = feats(z[f"fs__{r}"], gt)
    BAND[r] = {k: (float(np.nanpercentile(v[50:300], 5)), float(np.nanpercentile(v[50:300], 50)), float(np.nanpercentile(v[50:300], 95)))
               for k, v in GTF[r].items()}
KEYS = list(GTF[ROWS[0]])
DRIFT = {"S-05": [8, 12, 16, 17, 18, 19], "E-03": [16, 17, 18, 19]}


def u(r, k, x):
    lo, md, hi = BAND[r][k]
    half = max((hi - lo) / 2, 1e-6)
    return (x - md) / half          # |u| > 1 = outside the GT 5-95% band (approx., symmetric)


def out_of_band(r, k, x):
    lo, md, hi = BAND[r][k]
    return (x < lo) | (x > hi)


# flags
flags = {}
for r in DRIFT:
    flags[r] = {}
    for k in KEYS:
        ps = per_sec(FSF[r][k])
        rows_ = {}
        for t in DRIFT[r]:
            if t < len(ps) and np.isfinite(ps[t]):
                lo, md, hi = BAND[r][k]
                rows_[t] = {"val": round(float(ps[t]), 3), "out": bool(ps[t] < lo or ps[t] > hi), "u": round(float(u(r, k, ps[t])), 2)}
        # also: fraction of frames out of band in 12-20 s and in 2-12 s
        x = FSF[r][k]
        ob_late = float(np.nanmean(out_of_band(r, k, x[300:500]).astype(float)))
        ob_mid = float(np.nanmean(out_of_band(r, k, x[50:300]).astype(float)))
        flags[r][k] = {"band": [round(b, 3) for b in BAND[r][k]], "drift_secs": rows_, "frac_out_2-12": round(ob_mid, 2), "frac_out_12-20": round(ob_late, 2)}

# background: all 11 fs rows, fraction of frames out of their GT band, 2-12 vs 12-20 (GT itself: 10% by construction)
bg = {}
for k in KEYS:
    a = [float(np.nanmean(out_of_band(r, k, FSF[r][k][50:300]).astype(float))) for r in ROWS]
    b = [float(np.nanmean(out_of_band(r, k, FSF[r][k][300:500]).astype(float))) for r in ROWS]
    bg[k] = {"median_frac_out_2-12": round(float(np.median(a)), 2), "median_frac_out_12-20": round(float(np.median(b)), 2),
             "rows_mostly_out_12-20(>50%)": [r for r, v in zip(ROWS, b) if v > 0.5]}

json.dump({"band_note": "GT 2-12 s 5/50/95 pct of the clip; u = (x - median) / half band; out = outside 5-95",
           "flags": flags, "background_all_fs": bg}, open(f"{G}/late_drift.json", "w"), indent=1)

# ---------------------------------------------------------------- plot
PK = ["pitch (nose-eye_mid y)/eye_d", "yaw (nose-ear_mid x)/ear_d", "head x rel shoulder mid (sw0)", "head y rel shoulder mid (sw0)",
      "shoulder width / prefill", "shoulder mid x (canvas px)", "shoulder mid y (canvas px)", "shoulder tilt (deg)",
      "elbow L height above sh mid (sw0)", "elbow R height above sh mid (sw0)", "wrist L height above sh mid (sw0)", "wrist R height above sh mid (sw0)",
      "hand L to nose min dist (sw0, conf>0.3)", "hand R to nose min dist (sw0, conf>0.3)"]
FS_C, GT_C, BAND_C, BG_C, HL = "#eb6834", "#2a78d6", "#cfe0f5", "#b9b8b2", "#e34948"
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e"})
fig, axs = plt.subplots(len(PK), 3, figsize=(13, 2.0 * len(PK)), sharex=True)
ts = np.arange(20) + 0.5
for i, k in enumerate(PK):
    for j, r in enumerate(["S-05", "E-03"]):
        ax = axs[i, j]
        lo, md, hi = BAND[r][k]
        ax.axhspan(lo, hi, color=BAND_C, lw=0, label="GT 2-12 s 5-95%")
        ax.axvspan(0, 2, color="#e6e5e0", lw=0)
        for t in DRIFT[r]:
            ax.axvspan(t, t + 1, color=HL, alpha=0.10, lw=0)
        g = per_sec(GTF[r][k])
        ax.plot(np.arange(len(g)) + 0.5, g, color=GT_C, lw=1.2, label="GT (12 s)")
        f = per_sec(FSF[r][k])
        ax.plot(ts[:len(f)], f, color=FS_C, lw=1.6, marker="o", ms=2.5, label="fs generated")
        if i == 0:
            ax.set_title(f"{r} fs  (red = reported drift seconds)", fontsize=9)
        if j == 0:
            ax.set_ylabel(k, fontsize=7)
    ax = axs[i, 2]
    ax.axhspan(-1, 1, color=BAND_C, lw=0)
    ax.axvspan(0, 2, color="#e6e5e0", lw=0)
    for r in ROWS:
        f = u(r, k, per_sec(FSF[r][k]))
        hl = r in DRIFT
        ax.plot(ts[:len(f)], f, color=(FS_C if r == "S-05" else "#6250d6") if hl else BG_C, lw=1.5 if hl else 0.7, zorder=3 if hl else 1,
                label=r if hl else None)
    ax.set_ylim(-6, 6)
    if i == 0:
        ax.set_title("all 11 fs rows, in GT-band units (|u|>1 = outside)", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
for ax in axs[-1]:
    ax.set_xlabel("time (s)")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", fontsize=8, frameon=False)
fig.suptitle("Late drift scan: per-second features, fs generated vs the clip's GT band (grey = 2 s real prefill)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(f"{G}/late_S05_E03.png", dpi=80)

for r in DRIFT:
    print("==", r)
    for k in KEYS:
        fl = flags[r][k]
        outs = [t for t, v in fl["drift_secs"].items() if v["out"]]
        print(f"  {k:42s} band {fl['band']}  out@drift {outs}  u {[v['u'] for v in fl['drift_secs'].values()]}  frac_out 2-12 {fl['frac_out_2-12']} 12-20 {fl['frac_out_12-20']}")
print("== background (median over 11 fs rows, frac frames outside own GT band)")
for k in KEYS:
    print(f"  {k:42s} {bg[k]}")
