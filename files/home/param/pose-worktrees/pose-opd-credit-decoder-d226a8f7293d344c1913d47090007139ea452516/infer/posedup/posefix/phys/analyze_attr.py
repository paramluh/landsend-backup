#!/usr/bin/env python3
"""posefix PHYS-B: attribution analysis of replay_attr.py runs. Per (call, setting), seeds pooled, generated tokens only:
  kinematics   PHYS-A's phys.py tests (conf > 0.3), % frames failing (any / per test)
  confidence   body conf (joints 0-12 mean), % of the 55 joints below the raster floor 0.2, % drawn red (0.2-0.3)
  posture      phys_posture.geo: torso shear (hip / shoulder width) p50 / p95, torso lean p95, shoulder roll p95, head roll p95
  drift        the same per 10 s bin (250 frames at 25 fps)
  attribution  cond-model log-probs of the sampled codes (cb0: pose tower raw logits; light: cond-only decoder teacher-forced) for tokens
               with a kinematically flagged frame vs clean, and tail shares (p < 1e-3); REAL reference gt_c1.npz (real codes teacher-forced)
-> posefix/phys/attr_summary.json + stdout."""
import glob
import json
import os
import sys

SC = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/score"
sys.path.insert(0, SC)
import numpy as np  # noqa: E402
import score as S  # noqa: E402
import phys as PH  # noqa: E402
from phys_posture import geo  # noqa: E402
import hands as HD  # noqa: E402

R = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/phys/runs"
PERSON = "01_ab63d745"
ORDER = ["U", "C0", "Cb", "Cl", "Ltl", "Ltp", "K30"]
LABEL = {"U": "user preset (pose CFG 3.2 cb0+light, k34/k53, T1)", "C0": "pose CFG off", "Cb": "pose CFG cb0 only", "Cl": "pose CFG light only",
         "Ltl": "user + pose light temp 0.8", "Ltp": "user + pose cb0 temp 0.8", "K30": "user + pose light top-k 30"}
LN = np.log(1e-3)


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def decode(codes, N):
    ck, sk, _ = S.prep_codes(S._fix_codes(codes), N)
    kp, kp0 = S.to_keypoints(S.decode(ck), S.frame0_of(PERSON))
    k0 = kp0.numpy()
    return kp.numpy()[:, S.P.SEL].astype(np.float64), max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3), 2 * sk


def stats(km, fl):
    c = km[..., 2]
    roll, lean, shear, hroll, bconf = geo(km)
    return {"frames": int(len(km)), "fail_any_pct": r(100 * fl.any(1).mean(), 2),
            "body_conf": r(c[:, :13].mean()), "below_floor_pct": r(100 * (c < 0.2).mean(), 1), "red_pct": r(100 * ((c >= 0.2) & (c < 0.3)).mean(), 1),
            "shear_p50": r(np.median(shear), 2), "shear_p95": r(np.percentile(shear, 95), 2), "lean_p50": r(np.median(lean), 1),
            "lean_p95": r(np.percentile(lean, 95), 1), "sh_roll_p95": r(np.percentile(roll, 95), 1), "head_roll_p95": r(np.percentile(hroll, 95), 1)}


def main():
    groups = {}
    for f in sorted(glob.glob(f"{R}/*_*_s*.npz")):
        z = np.load(f)
        key = os.path.basename(f).split("_")[0]
        st, N = str(z["setting"]), int(z["N"])
        k55, sw, lo = decode(z["codes"].astype(np.int64), N)
        fl = PH.frames(k55, sw)[1]
        km, flm = k55[lo:], fl[lo:]
        g = groups.setdefault((key, st), {"km": [], "fl": [], "bins": [], "lp0": [], "lpl": [], "lmn": [], "tf": [], "runs": 0, "rep": [], "gpm": []})
        gen = z["codes"].astype(np.int64)[N:, 0]
        g["rep"].append(100 * float((gen[1:] == gen[:-1]).mean()))
        g["gpm"].append(HD.metrics(km, sw)["gestures_per_min"])
        g["runs"] += 1
        g["km"].append(km); g["fl"].append(flm)
        g["bins"].append([stats(km[i:i + 250], flm[i:i + 250]) for i in range(0, len(km), 250) if len(km[i:i + 250]) >= 125])
        lp0, lpl = z["lp0"][N:], z["lpl"][N:]
        n = min(len(lp0), len(flm) // 2)
        if np.isfinite(lpl[:n]).all() and n:
            g["lp0"].append(lp0[:n]); g["lpl"].append(lpl[:n].sum(1)); g["lmn"].append(lpl[:n].min(1))
            g["tf"].append(flm[:2 * n].reshape(n, 2, -1).any(1).any(1))
    out = {}
    keys = sorted({k for k, _ in groups}, key=lambda k: ["7b5a", "b62a", "b62aL"].index(k) if k in ("7b5a", "b62a", "b62aL") else 9)
    for key in keys:
        for st in [s for s in ORDER if (key, s) in groups]:
            g = groups[(key, st)]
            d = {"label": LABEL[st], "runs": g["runs"], **stats(np.concatenate(g["km"]), np.concatenate(g["fl"])),
                 "cb0_repeat_pct": r(np.mean(g["rep"]), 1), "gestures_per_min": r(np.mean(g["gpm"]), 1)}
            nb = min(len(b) for b in g["bins"])
            d["bins_10s"] = [{k: r(np.mean([b[i][k] for b in g["bins"]]), 3) for k in ("body_conf", "below_floor_pct", "shear_p50", "lean_p50", "lean_p95",
                                                                                        "fail_any_pct")} for i in range(nb)]
            if g["lp0"]:
                lp0, lpl, lmn, tf = (np.concatenate(g[k]) for k in ("lp0", "lpl", "lmn", "tf"))
                d["cb0_lp_mean"], d["light_lp_sum_mean"] = r(lp0.mean()), r(lpl.mean(), 2)
                d["cb0_tail_pct"], d["light_any_tail_pct"] = r(100 * (lp0 < LN).mean(), 2), r(100 * (lmn < LN).mean(), 2)
                if tf.any() and (~tf).any():
                    d["flagged_vs_clean"] = {"n_flagged": int(tf.sum()), "cb0_lp": [r(lp0[tf].mean()), r(lp0[~tf].mean())],
                                             "light_lp_sum": [r(lpl[tf].mean(), 2), r(lpl[~tf].mean(), 2)],
                                             "light_any_tail_pct": [r(100 * (lmn[tf] < LN).mean(), 1), r(100 * (lmn[~tf] < LN).mean(), 1)]}
            out[f"{key}/{st}"] = d
            print(f"{key}/{st}", json.dumps({k: v for k, v in d.items() if k != "bins_10s"}), flush=True)
            print("   bins", json.dumps([[b["body_conf"], b["below_floor_pct"], b["shear_p50"], b["lean_p95"]] for b in d["bins_10s"]]), flush=True)
    g = f"{R}/gt_c1.npz"
    if os.path.exists(g):
        z = np.load(g)
        lp0, lpl = z["lp0"], z["lpl"]
        out["REAL_gt_c1"] = {"tokens": int(len(lp0)), "cb0_lp_mean": r(lp0.mean()), "light_lp_sum_mean": r(lpl.sum(1).mean(), 2),
                             "cb0_tail_pct": r(100 * (lp0 < LN).mean(), 2), "light_any_tail_pct": r(100 * (lpl.min(1) < LN).mean(), 2)}
        print("REAL", json.dumps(out["REAL_gt_c1"]), flush=True)
    json.dump(out, open("/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/phys/attr_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
