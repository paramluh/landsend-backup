"""H1-H4 head-geometry metrics on decoded keypoints (kp.npz from decode_all.py). 25 fps; prefill = 0-2 s (frames 0-49).
Conditions: GT (package tokens, 12 s), PF (0-2 s prefill inside the fs sample), fs, d1, d2, d3 (20 s)."""
import json
import sys

import numpy as np

G = "/data/home/amartyavarma/ppm/gaze-drift/geom"
FPS, PF = 25, 50
CLIP = {"S-05": "10_27286bdd", "S-13": "06_74807fe0", "S-15": "10_27286bdd", "S-20": "19_47473a46", "E-03": "03_b437950e",
        "E-05": "05_e25c6859", "E-09": "09_d9ef3c94", "E-12": "12_72cb45d3", "E-13": "13_956c09af", "E-15": "15_d1e1322c",
        "E-18": "18_1ffa9083"}
ROWS = list(CLIP)
GEN = ["fs", "d1", "d2", "d3"]
BINS = {"0-2": (0, 50), "2-6": (50, 150), "6-12": (150, 300), "12-20": (300, 500), "2-12": (50, 300)}
z = np.load(f"{G}/kp.npz")
KP = {}
for r in ROWS:
    for c in GEN:
        KP[(c, r)] = z[f"{c}__{r}"]
    KP[("GT", r)] = z[f"gtpkg__{CLIP[r]}"]
HEAD = [0, 1, 2, 3, 4]


def nrm(v):
    return np.linalg.norm(v, axis=-1)


def series(kp):
    """per-frame signals [T] for one sample; scale = prefill-median inter-eye distance (constant per sample)."""
    xy, cf = kp[..., :2], kp[..., 2]
    sh = (xy[:, 5] + xy[:, 6]) / 2
    eye_d, ear_d = nrm(xy[:, 1] - xy[:, 2]), nrm(xy[:, 3] - xy[:, 4])
    sw = nrm(xy[:, 5] - xy[:, 6])
    s_eye, s_ear, s_sw = np.median(eye_d[:PF]), np.median(ear_d[:PF]), np.median(sw[:PF])
    rel = (xy[:, HEAD] - sh[:, None]) / s_eye                                   # [T, 5, 2] head pts rel shoulder mid, eye units
    eye_mid, ear_mid = (xy[:, 1] + xy[:, 2]) / 2, (xy[:, 3] + xy[:, 4]) / 2
    nose = xy[:, 0]
    S = {}
    S["rel"] = rel
    S["yaw_ear"] = (nose[:, 0] - ear_mid[:, 0]) / ear_d
    S["yaw_eye"] = (nose[:, 0] - eye_mid[:, 0]) / eye_d
    S["pitch"] = (nose[:, 1] - eye_mid[:, 1]) / eye_d
    d = xy[:, 1] - xy[:, 2]
    S["roll"] = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    S["roll"] = S["roll"] - 360 * np.round((S["roll"] - np.median(S["roll"][:PF])) / 360)   # unwrap around prefill
    S["headpos_x"] = (nose[:, 0] - sh[:, 0]) / s_sw
    S["headpos_y"] = (nose[:, 1] - sh[:, 1]) / s_sw
    # H3
    lN, rN = nrm(xy[:, 1] - nose), nrm(xy[:, 2] - nose)
    S["eye_asym"] = (lN - rN) / (lN + rN)
    S["eyemid_x_face"] = (eye_mid[:, 0] - ear_mid[:, 0]) / ear_d
    S["eyemid_y_face"] = (eye_mid[:, 1] - ear_mid[:, 1]) / ear_d
    S["eye_ear_ratio"] = eye_d / ear_d
    S["ear_d_norm"] = ear_d / s_ear
    # absolute canvas motion (what the renderer sees), in prefill shoulder widths
    S["absnose_x"], S["absnose_y"] = nose[:, 0] / s_sw, nose[:, 1] / s_sw
    S["shmid_x"], S["shmid_y"] = sh[:, 0] / s_sw, sh[:, 1] / s_sw
    S["sw_ratio"] = sw / s_sw
    # H4
    S["conf"] = cf[:, HEAD]
    S["scales"] = (s_eye, s_ear, s_sw)
    return S


ORIENT = ["yaw_ear", "yaw_eye", "pitch", "roll", "headpos_x", "headpos_y", "eye_asym", "eyemid_x_face", "eyemid_y_face", "eye_ear_ratio",
          "absnose_x", "absnose_y", "shmid_x", "shmid_y", "sw_ratio"]
SER = {k: series(v) for k, v in KP.items()}


def jitter(S, a, b):
    rel = S["rel"]
    v = nrm(np.diff(rel, axis=0)).mean(1)          # [T-1] mean |Δ| over 5 head pts (eye units / frame)
    acc = nrm(np.diff(rel, 2, axis=0)).mean(1)
    v12 = nrm(rel[2::2] - rel[:-2:2]).mean(1)      # 12.5 Hz step (token rate)
    # within-token (2i -> 2i+1) vs across-token (2i+1 -> 2i+2) displacement
    win = nrm(rel[1::2] - rel[0::2][:len(rel[1::2])]).mean(1)
    acr = nrm(rel[2::2] - rel[1::2][:len(rel[2::2])]).mean(1)
    vv = np.diff(rel, axis=0).reshape(len(rel) - 1, -1)
    o = {}
    sl = slice(a, b - 1)
    o["jit"] = float(v[sl].mean())
    o["acc"] = float(acc[a:b - 2].mean())
    o["jit12"] = float(v12[a // 2:(b - 2) // 2].mean())
    o["zigzag"] = float(2 * v[sl].mean() / max(v12[a // 2:(b - 2) // 2].mean(), 1e-9))     # 1 = smooth/straight, >1 = back-and-forth
    o["within_tok"] = float(win[a // 2:b // 2].mean())
    o["across_tok"] = float(acr[a // 2:(b - 1) // 2].mean())
    x = vv[sl]
    x = x - x.mean(0)
    o["vel_ac1"] = float((x[1:] * x[:-1]).sum() / max((x * x).sum(), 1e-12))    # lag-1 velocity autocorr (<0 = zigzag)
    # nose-only jitter rel shoulders, eye units, and fraction of frames with |Δ| > 0.1 eye-dist
    o["jit_nose"] = float(nrm(np.diff(rel[:, 0], axis=0))[sl].mean())
    o["jit_big_frac"] = float((v[sl] > 0.1).mean())
    # spectral band power of head motion rel shoulders (5 pts, x+y), per band, eye units^2
    seg = rel[a:b].reshape(b - a, -1)
    seg = seg - seg.mean(0)
    P = (np.abs(np.fft.rfft(seg * np.hanning(len(seg))[:, None], axis=0)) ** 2).sum(1) / len(seg)
    fq = np.fft.rfftfreq(len(seg), 1 / FPS)
    for lo, hi in [(0.1, 0.5), (0.5, 2), (2, 5), (5, 12.6)]:
        o[f"band_{lo}-{hi}Hz"] = float(P[(fq >= lo) & (fq < hi)].sum())
    return o


def orient_stats(S, a, b, key):
    x = S[key]
    m0 = x[:PF].mean()
    dev = np.abs(x - m0)
    o = {"mean_absdev": float(dev[a:b].mean()), "max_absdev": float(dev[a:b].max()), "std": float(x[a:b].std()),
         "mean": float(x[a:b].mean())}
    return o


def msd_exponent(x, a, b, lags=(5, 10, 20, 40, 75, 125)):
    seg = x[a:b]
    out = []
    for L in lags:
        if L < len(seg) // 2:
            out.append((L, float(np.mean((seg[L:] - seg[:-L]) ** 2))))
    if len(out) < 3:
        return None, out
    lx, ly = np.log([o[0] for o in out]), np.log([max(o[1], 1e-12) for o in out])
    return float(np.polyfit(lx, ly, 1)[0]), out


def acf_time(x, a, b):
    seg = x[a:b] - x[a:b].mean()
    den = (seg * seg).sum()
    if den <= 0:
        return 0.0
    for L in range(1, len(seg) // 2):
        if (seg[L:] * seg[:-L]).sum() / den < 1 / np.e:
            return L / FPS
    return len(seg) / 2 / FPS


def conf_stats(S, a, b):
    c = S["conf"][a:b]
    ears = c[:, 3:5]
    flips = 0
    for j in range(2):
        s = ears[:, j] > 0.5
        flips += int((s[1:] != s[:-1]).sum())
    return {"conf_head": float(c.mean()), "conf_nose": float(c[:, 0].mean()), "conf_eyes": float(c[:, 1:3].mean()),
            "conf_ears": float(ears.mean()), "conf_ear_min": float(ears.min(1).mean()),
            "ear_conf_asym": float(np.abs(ears[:, 0] - ears[:, 1]).mean()),
            "frac_any_lt03": float((c.min(1) < 0.3).mean()), "frac_any_lt05": float((c.min(1) < 0.5).mean()),
            "frac_ear_lt03": float((ears.min(1) < 0.3).mean()), "frac_ear_lt05": float((ears.min(1) < 0.5).mean()),
            "ear_flip_per_s": float(flips / ((b - a) / FPS))}


# ------------------------------------------------------------------ per sample, per bin metrics
M = {}   # M[(cond,row,bin)] = dict
for (c, r), S in SER.items():
    T = len(S["yaw_ear"])
    for bn, (a, b) in BINS.items():
        if b > T:
            continue
        o = {}
        o.update({f"H1_{k}": v for k, v in jitter(S, a, b).items()})
        for key in ORIENT:
            for k2, v in orient_stats(S, a, b, key).items():
                o[f"{key}_{k2}"] = v
        o.update({f"H4_{k}": v for k, v in conf_stats(S, a, b).items()})
        M[(c, r, bn)] = o
    # full-length random-walk stats on the post-prefill part (GT: 2-12 s; gen: 2-12 s matched and 2-20 s)
    for wn, (a, b) in {"2-12": (50, 300), "2-20": (50, 500)}.items():
        if b > T:
            continue
        o = M.setdefault((c, r, "rw" + wn), {})
        for key in ["yaw_ear", "yaw_eye", "pitch", "roll", "headpos_x", "headpos_y", "eyemid_x_face", "eye_asym"]:
            ex, _ = msd_exponent(SER[(c, r)][key], a, b)
            o[f"{key}_msd_exp"] = ex
            o[f"{key}_acf_s"] = acf_time(SER[(c, r)][key], a, b)
            x = SER[(c, r)][key]
            dev = np.abs(x - x[:PF].mean())
            # growth: slope of |dev| vs time (per s), first half vs second half of window
            t = np.arange(a, b) / FPS
            o[f"{key}_dev_slope"] = float(np.polyfit(t, dev[a:b], 1)[0])
            # return toward camera: fraction of time spent closer to prefill mean than at the window's max excursion / 2
            mx = dev[a:b].max()
            imx = a + int(dev[a:b].argmax())
            o[f"{key}_return_after_max"] = float(dev[imx:b].min() / max(mx, 1e-9)) if imx < b - 1 else 1.0
            o[f"{key}_frac_within_prefill_range"] = float(((x[a:b] >= x[:PF].min()) & (x[a:b] <= x[:PF].max())).mean())

# prefill condition: 0-2 s of the fs sample
for r in ROWS:
    M[("PF", r, "0-2")] = M[("fs", r, "0-2")]

# thresholds from GT for "fraction beyond threshold": pooled GT 90th pct of |dev| over 2-12 s
THR = {}
for key in ORIENT:
    pool = np.concatenate([np.abs(SER[("GT", r)][key][50:300] - SER[("GT", r)][key][:PF].mean()) for r in ROWS])
    THR[key] = float(np.percentile(pool, 90))
for (c, r), S in SER.items():
    T = len(S["yaw_ear"])
    for bn, (a, b) in BINS.items():
        if b > T or (c, r, bn) not in M:
            continue
        for key in ORIENT:
            x = S[key]
            M[(c, r, bn)][f"{key}_frac_beyond_gt90"] = float((np.abs(x[a:b] - x[:PF].mean()) > THR[key]).mean())

# ------------------------------------------------------------------ summaries
CONDS = ["GT", "PF", "fs", "d1", "d2", "d3"]


def med(c, bn, k):
    v = [M[(c, r, bn)][k] for r in ROWS if (c, r, bn) in M and M[(c, r, bn)].get(k) is not None]
    return float(np.median(v)) if v else None


def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None]).sum() - (a[:, None] < b[None]).sum()) / (len(a) * len(b)))


keys = sorted(M[("fs", ROWS[0], "2-12")].keys())
summary = {}
for k in keys:
    s = {}
    for bn in ["0-2", "2-6", "6-12", "12-20", "2-12"]:
        s[bn] = {c: med(c, bn, k) for c in CONDS if not (c == "PF" and bn != "0-2") and not (c == "GT" and bn == "12-20")}
    # matched 2-12 s: per-row ratio gen/GT (GT unique per clip; S-05/S-15 share), effect size gen vs GT (unique GT clips)
    gt_rows = {CLIP[r]: M[("GT", r, "2-12")][k] for r in ROWS}
    gtv = list(gt_rows.values())
    s["ratio_gen_over_gt_2-12"] = {}
    s["cliff_gen_vs_gt_2-12"] = {}
    for c in GEN:
        rat = [M[(c, r, "2-12")][k] / gt_rows[CLIP[r]] for r in ROWS if abs(gt_rows[CLIP[r]]) > 1e-9]
        s["ratio_gen_over_gt_2-12"][c] = float(np.median(rat)) if rat else None
        s["cliff_gen_vs_gt_2-12"][c] = cliff([M[(c, r, "2-12")][k] for r in ROWS], gtv)
    allgen = [M[(c, r, "2-12")][k] for c in GEN for r in ROWS]
    s["cliff_allgen_vs_gt_2-12"] = cliff(allgen, gtv)
    # Cohen-style d on log values where positive
    a_, b_ = np.asarray(allgen, float), np.asarray(gtv, float)
    sd = np.sqrt((a_.var(ddof=1) + b_.var(ddof=1)) / 2)
    s["cohen_d_allgen_vs_gt_2-12"] = float((a_.mean() - b_.mean()) / sd) if sd > 0 else None
    summary[k] = s

rwkeys = sorted(M[("fs", ROWS[0], "rw2-12")].keys())
rw = {}
for k in rwkeys:
    s = {}
    for wn in ["rw2-12", "rw2-20"]:
        s[wn] = {c: med(c, wn, k) for c in ["GT"] + GEN if not (c == "GT" and wn == "rw2-20")}
    gtv = list({CLIP[r]: M[("GT", r, "rw2-12")][k] for r in ROWS}.values())
    allgen = [M[(c, r, "rw2-12")][k] for c in GEN for r in ROWS]
    if None not in gtv and None not in allgen:
        s["cliff_allgen_vs_gt_2-12"] = cliff(allgen, gtv)
    rw[k] = s

ranked = sorted([(k, v["cliff_allgen_vs_gt_2-12"], v["ratio_gen_over_gt_2-12"]["fs"], v["cohen_d_allgen_vs_gt_2-12"]) for k, v in summary.items()],
                key=lambda t: -abs(t[1]) - 0.01 * abs(t[3] or 0))
ranked_rw = sorted([(k, v.get("cliff_allgen_vs_gt_2-12")) for k, v in rw.items() if v.get("cliff_allgen_vs_gt_2-12") is not None],
                   key=lambda t: -abs(t[1]))

# continuous curves (0.5 s bins): median + IQR over rows of |dev from prefill mean| for the orientation keys, jitter, conf
curves = {}
EDGES = np.arange(0, 501, 12.5).astype(int)
for key in ORIENT + ["jit", "conf_head", "conf_ears"]:
    curves[key] = {}
    for c in ["GT"] + GEN:
        rows = []
        for r in ROWS:
            S = SER[(c, r)]
            if key == "jit":
                x = np.r_[0, nrm(np.diff(S["rel"], axis=0)).mean(1)]
            elif key == "conf_head":
                x = S["conf"].mean(1)
            elif key == "conf_ears":
                x = S["conf"][:, 3:5].mean(1)
            else:
                x = np.abs(S[key] - S[key][:PF].mean())
            rows.append([float(x[a:b].mean()) if b <= len(x) else np.nan for a, b in zip(EDGES[:-1], EDGES[1:])])
        A = np.array(rows)
        curves[key][c] = {"t": [float((a + b) / 2 / FPS) for a, b in zip(EDGES[:-1], EDGES[1:])],
                          "med": np.nanmedian(A, 0).tolist() if np.isfinite(A).any() else None,
                          "q25": np.nanpercentile(A, 25, 0).tolist(), "q75": np.nanpercentile(A, 75, 0).tolist()}

out = {"note": "units: head jitter in prefill inter-eye distances per 25-fps frame, rel. shoulder midpoint; yaw_ear=(nose_x-ear_mid_x)/ear_dist, "
               "yaw_eye=(nose_x-eye_mid_x)/eye_dist, pitch=(nose_y-eye_mid_y)/eye_dist, roll=eye-line deg, headpos=(nose-shoulder_mid)/prefill shoulder width; "
               "_mean_absdev = mean |x - prefill mean|; conditions GT=package tokens (12 s, same as rendered GT), PF=0-2 s prefill of fs sample.",
       "gt_thresholds_90pct_absdev_2-12": THR, "summary": summary, "random_walk": rw, "ranked_separation_2-12": ranked,
       "ranked_rw_separation": ranked_rw, "curves": curves,
       "per_sample": {f"{c}/{r}/{bn}": v for (c, r, bn), v in M.items()}}
json.dump(out, open(f"{G}/results.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ short print
P = ["H1_jit", "H1_acc", "H1_jit12", "H1_zigzag", "H1_vel_ac1", "H1_within_tok", "H1_across_tok", "H1_jit_big_frac",
     "yaw_ear_mean_absdev", "yaw_eye_mean_absdev", "pitch_mean_absdev", "roll_mean_absdev", "headpos_x_mean_absdev", "headpos_y_mean_absdev",
     "yaw_ear_max_absdev", "yaw_ear_frac_beyond_gt90",
     "eye_asym_mean_absdev", "eyemid_x_face_mean_absdev", "eyemid_y_face_mean_absdev", "eye_ear_ratio_mean_absdev",
     "H4_conf_head", "H4_conf_ears", "H4_frac_any_lt03", "H4_frac_any_lt05", "H4_ear_flip_per_s", "H4_ear_conf_asym"]
for k in P:
    s = summary[k]
    print(f"{k:28s} " + " | ".join(f"{bn}: " + " ".join(f"{c}={v:.3g}" for c, v in s[bn].items() if v is not None) for bn in ["0-2", "2-12", "12-20"])
          + f" | ratio_fs={s['ratio_gen_over_gt_2-12']['fs']:.2f} cliff={s['cliff_allgen_vs_gt_2-12']:+.2f}")
print("RW")
for k in ["yaw_ear_msd_exp", "yaw_eye_msd_exp", "pitch_msd_exp", "roll_msd_exp", "headpos_x_msd_exp", "yaw_ear_acf_s", "yaw_ear_dev_slope",
          "yaw_ear_return_after_max", "yaw_ear_frac_within_prefill_range", "headpos_x_dev_slope", "pitch_dev_slope"]:
    print(k, rw[k])
print("TOP separation:", [(k, round(c, 2), round(r, 2) if r else r) for k, c, r, d in ranked[:12]])
print("TOP rw separation:", ranked_rw[:8])
print("THR", THR)
