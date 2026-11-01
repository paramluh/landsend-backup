"""Coordinator follow-ups: (1) body/hand motion energy, head-torso coupling, stillness; (2) 12-20 s window;
(3) signed static yaw/pitch offset vs prefill per row; (4) monotonic yaw trend per row. Adds 'extras' to results.json."""
import json

import numpy as np

G = "/data/home/amartyavarma/ppm/gaze-drift/geom"
FPS, PF = 25, 50
CLIP = {"S-05": "10_27286bdd", "S-13": "06_74807fe0", "S-15": "10_27286bdd", "S-20": "19_47473a46", "E-03": "03_b437950e",
        "E-05": "05_e25c6859", "E-09": "09_d9ef3c94", "E-12": "12_72cb45d3", "E-13": "13_956c09af", "E-15": "15_d1e1322c",
        "E-18": "18_1ffa9083"}
ROWS = list(CLIP)
GEN = ["fs", "d1", "d2", "d3"]
CONDS = ["GT"] + GEN
WIN = {"0-2": (0, 50), "2-12": (50, 300), "12-20": (300, 500)}
z = np.load(f"{G}/kp.npz")
KP = {(c, r): z[f"{c}__{r}"] for c in GEN for r in ROWS}
KP.update({("GT", r): z[f"gtpkg__{CLIP[r]}"] for r in ROWS})
nrm = lambda v: np.linalg.norm(v, axis=-1)  # noqa: E731


def rank(x):
    o = np.argsort(x)
    rk = np.empty(len(x))
    rk[o] = np.arange(len(x))
    return rk


def spearman(x, y):
    a, b = rank(x), rank(y)
    return float(np.corrcoef(a, b)[0, 1])


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


per = {}
for (c, r), kp in KP.items():
    xy = kp[..., :2]
    T = len(xy)
    sh = (xy[:, 5] + xy[:, 6]) / 2
    sw0 = np.median(nrm(xy[:PF, 5] - xy[:PF, 6]))
    eye0 = np.median(nrm(xy[:PF, 1] - xy[:PF, 2]))
    head_rel = (xy[:, :5] - sh[:, None]) / eye0
    head_spd = nrm(np.diff(head_rel, axis=0)).mean(1)                       # eye dists / frame
    torso_spd = nrm(np.diff(sh, axis=0)) / sw0                              # shoulder mid, shoulder widths / frame
    body_spd = nrm(np.diff(xy[:, 5:13], axis=0)).mean(1) / sw0             # shoulders, elbows, wrists, hips (abs)
    arm_spd = nrm(np.diff(xy[:, 7:11] - sh[:, None], axis=0)).mean(1) / sw0  # elbows+wrists rel shoulder mid
    hand_spd = nrm(np.diff(xy[:, 13:55] - xy[:, [9] * 21 + [10] * 21], axis=0)).mean(1) / sw0  # fingers rel own wrist
    hand_abs = nrm(np.diff(xy[:, 13:55], axis=0)).mean(1) / sw0
    nose_v = np.diff(xy[:, 0], axis=0) / sw0
    sh_v = np.diff(sh, axis=0) / sw0
    ear_mid = (xy[:, 3] + xy[:, 4]) / 2
    ear_d = nrm(xy[:, 3] - xy[:, 4])
    eye_mid = (xy[:, 1] + xy[:, 2]) / 2
    eye_d = nrm(xy[:, 1] - xy[:, 2])
    yaw = (xy[:, 0, 0] - ear_mid[:, 0]) / ear_d
    yaw_eye = (xy[:, 0, 0] - eye_mid[:, 0]) / eye_d
    pitch = (xy[:, 0, 1] - eye_mid[:, 1]) / eye_d
    for wn, (a, b) in WIN.items():
        if b > T:
            continue
        s = slice(a, b - 1)
        o = {}
        o["head_spd"] = float(head_spd[s].mean())
        o["torso_spd"] = float(torso_spd[s].mean())
        o["body_spd"] = float(body_spd[s].mean())
        o["arm_spd"] = float(arm_spd[s].mean())
        o["hand_finger_spd"] = float(hand_spd[s].mean())
        o["hand_abs_spd"] = float(hand_abs[s].mean())
        o["still_frac_005"] = float((head_spd[s] < 0.005).mean())
        o["still_frac_010"] = float((head_spd[s] < 0.010).mean())
        o["longest_still_s_005"] = longest_run(head_spd[s] < 0.005) / FPS
        o["longest_still_s_010"] = longest_run(head_spd[s] < 0.010) / FPS
        # head-torso coupling: corr of nose and shoulder-mid velocity (x and y), and of head-rel speed vs torso speed
        o["couple_vx"] = float(np.corrcoef(nose_v[s, 0], sh_v[s, 0])[0, 1])
        o["couple_vy"] = float(np.corrcoef(nose_v[s, 1], sh_v[s, 1])[0, 1])
        o["couple_spd"] = float(np.corrcoef(head_spd[s], torso_spd[s])[0, 1])
        # (3) signed static offsets vs prefill mean
        for nm, x in [("yaw_ear", yaw), ("yaw_eye", yaw_eye), ("pitch", pitch)]:
            o[f"{nm}_offset"] = float(x[a:b].mean() - x[:PF].mean())
            o[f"{nm}_std"] = float(x[a:b].std())
        per[f"{c}/{r}/{wn}"] = o
    # (4) monotonic trend over the whole post-prefill part
    b = T
    t = np.arange(PF, b) / FPS
    for nm, x in [("yaw_ear", yaw), ("pitch", pitch)]:
        per[f"{c}/{r}/trend"] = per.get(f"{c}/{r}/trend", {})
        per[f"{c}/{r}/trend"][f"{nm}_spearman"] = spearman(t, x[PF:b])
        per[f"{c}/{r}/trend"][f"{nm}_slope_per10s"] = float(np.polyfit(t, x[PF:b], 1)[0] * 10)
        per[f"{c}/{r}/trend"][f"{nm}_range"] = float(x[PF:b].max() - x[PF:b].min())

# GT reference std per clip (2-12 s) for normalising offsets
gt_std = {r: per[f"GT/{r}/2-12"]["yaw_ear_std"] for r in ROWS}
gt_pstd = {r: per[f"GT/{r}/2-12"]["pitch_std"] for r in ROWS}

summ = {}
keys = list(per[f"fs/{ROWS[0]}/2-12"].keys())
for k in keys:
    summ[k] = {}
    for wn in WIN:
        summ[k][wn] = {}
        for c in CONDS:
            v = [per[f"{c}/{r}/{wn}"][k] for r in ROWS if f"{c}/{r}/{wn}" in per]
            if v:
                summ[k][wn][c] = float(np.median(v))
    gt = [per[f"GT/{r}/2-12"][k] for r in ROWS]
    summ[k]["ratio_gen_over_gt_2-12"] = {c: float(np.median([per[f"{c}/{r}/2-12"][k] / gt[i] for i, r in enumerate(ROWS) if abs(gt[i]) > 1e-9]))
                                        for c in GEN}

offset_rows = {}
for r in ROWS:
    offset_rows[r] = {}
    for c in CONDS:
        for wn in ["2-12", "12-20"]:
            key = f"{c}/{r}/{wn}"
            if key in per:
                offset_rows[r][f"{c}_{wn}"] = {"yaw": round(per[key]["yaw_ear_offset"], 4), "yaw_in_gtstd": round(per[key]["yaw_ear_offset"] / gt_std[r], 2),
                                               "pitch": round(per[key]["pitch_offset"], 4), "pitch_in_gtstd": round(per[key]["pitch_offset"] / gt_pstd[r], 2)}
abs_off = {}
for c in CONDS:
    for wn in ["2-12", "12-20"]:
        v = [abs(per[f"{c}/{r}/{wn}"]["yaw_ear_offset"]) / gt_std[r] for r in ROWS if f"{c}/{r}/{wn}" in per]
        p = [abs(per[f"{c}/{r}/{wn}"]["pitch_offset"]) / gt_pstd[r] for r in ROWS if f"{c}/{r}/{wn}" in per]
        if v:
            abs_off[f"{c}_{wn}"] = {"med_abs_yaw_offset_in_gtstd": float(np.median(v)), "n_rows_gt_1std": int(sum(x > 1 for x in v)),
                                    "med_abs_yaw_offset": float(np.median([abs(per[f"{c}/{r}/{wn}"]["yaw_ear_offset"]) for r in ROWS])),
                                    "med_abs_pitch_offset_in_gtstd": float(np.median(p)), "n_rows_pitch_gt_1std": int(sum(x > 1 for x in p))}
trend = {}
for c in CONDS:
    rho = [per[f"{c}/{r}/trend"]["yaw_ear_spearman"] for r in ROWS]
    sl = [per[f"{c}/{r}/trend"]["yaw_ear_slope_per10s"] for r in ROWS]
    rng = [per[f"{c}/{r}/trend"]["yaw_ear_range"] for r in ROWS]
    mono = [r for r, a, s in zip(ROWS, rho, sl) if abs(a) > 0.5 and abs(s) > gt_std[r]]
    trend[c] = {"med_abs_spearman": float(np.median(np.abs(rho))), "n_abs_rho_gt_0.5": int(sum(abs(a) > 0.5 for a in rho)),
                "monotonic_rows(|rho|>0.5 & |slope/10s|>GT std)": mono, "per_row_rho": dict(zip(ROWS, [round(a, 2) for a in rho])),
                "per_row_slope_per10s": dict(zip(ROWS, [round(a, 3) for a in sl])), "med_range": float(np.median(rng))}

R = json.load(open(f"{G}/results.json"))
R["extras"] = {"summary": summ, "yaw_pitch_offset_per_row": offset_rows, "abs_offset_summary": abs_off, "yaw_trend": trend,
               "per_sample": per,
               "note": "speeds per 25-fps frame: head_spd in prefill eye dists (rel shoulder mid); torso/body/arm/hand in prefill shoulder widths. "
                       "offset = window mean - prefill mean (signed); *_in_gtstd = offset / GT clip's own 2-12 s std. trend over 2 s..end."}
json.dump(R, open(f"{G}/results.json", "w"), indent=1)

for k in ["head_spd", "torso_spd", "body_spd", "arm_spd", "hand_finger_spd", "hand_abs_spd", "still_frac_005", "still_frac_010",
          "longest_still_s_005", "longest_still_s_010", "couple_vx", "couple_vy", "couple_spd"]:
    print(f"{k:20s}", {wn: {c: round(v, 4) for c, v in summ[k][wn].items()} for wn in WIN}, "ratio", {c: round(v, 2) for c, v in summ[k]["ratio_gen_over_gt_2-12"].items()})
print("ABS OFFSET", json.dumps(abs_off))
for c in CONDS:
    print("TREND", c, {k: v for k, v in trend[c].items() if k not in ("per_row_slope_per10s",)})
for r in ROWS:
    print("OFF", r, {k: (v["yaw"], v["yaw_in_gtstd"], v["pitch_in_gtstd"]) for k, v in offset_rows[r].items()})
