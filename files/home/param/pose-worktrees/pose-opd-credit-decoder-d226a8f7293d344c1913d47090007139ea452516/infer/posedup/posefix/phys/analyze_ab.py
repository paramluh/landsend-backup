#!/usr/bin/env python3
"""posefix PHYS-B: A/B table for ab_replay.py runs (generated frames only), per arm (mean over runs) + per 10 s drift, and the REAL Aloe
reference rows from the same scorers (held-out Aloe B tracks in Person 01's frame: score/work/phys_real_k55.npz, PHYS-A's cache).
  physicality  body conf (joints 0-12), % joints < 0.2 (raster floor), % red (0.2-0.3), torso lean p95, head roll p95 (phys_posture.geo)
  motion       cb0 repeat % (consecutive generated tokens), gestures / min and hands moving % (score/hands.py)
-> posefix/phys/ab_summary.json + stdout."""
import glob
import json
import os
import sys

SC = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/score"
sys.path.insert(0, SC)
import numpy as np  # noqa: E402
import score as S  # noqa: E402
import hands as HD  # noqa: E402
from phys_posture import geo  # noqa: E402

R = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/phys/runs_ab"
PERSON = "01_ab63d745"
ARMS = {"U": "user preset (pose CFG 3.2 cb0+light, pose k34, light k53, T1)", "a": "U with CFG on cb0 only (pose_cfg_light 0)",
        "b": "U with pose CFG off + pose light top-k 151"}


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def phys(km):
    c = km[..., 2]
    roll, lean, shear, hroll, _ = geo(km)
    return {"body_conf": float(c[:, :13].mean()), "below_floor_pct": 100 * float((c < 0.2).mean()), "red_pct": 100 * float(((c >= 0.2) & (c < 0.3)).mean()),
            "lean_p95": float(np.percentile(lean, 95)), "head_roll_p95": float(np.percentile(hroll, 95)), "shear_p50": float(np.median(shear))}


def metrics(km, sw, cb0):
    d = phys(km)
    h = HD.metrics(km, sw)
    d.update(cb0_repeat_pct=100 * float((cb0[1:] == cb0[:-1]).mean()), gestures_per_min=h["gestures_per_min"], hands_moving_pct=h["hands_moving_pct"])
    return d


def mean_rows(rows):
    return {k: r(np.mean([x[k] for x in rows]), 3) for k in rows[0]}


def main():
    arms = {}
    only1 = len(sys.argv) > 1 and sys.argv[1] == "seed1"                # the first seed of each mic only
    SEED1 = {"7b5a": 14871328, "b62a": 57851, "m1086": 1086, "c2": 2002}
    for f in sorted(glob.glob(f"{R}/*_*_s*.npz")):
        mic, arm = os.path.basename(f).split("_")[:2]
        if only1 and not f.endswith(f"_s{SEED1[mic]}.npz"):
            continue
        z = np.load(f)
        N = int(z["N"])
        codes = z["codes"].astype(np.int64)
        ck, sk, _ = S.prep_codes(S._fix_codes(codes), N)
        kp, kp0 = S.to_keypoints(S.decode(ck), S.frame0_of(PERSON))
        k0 = kp0.numpy()
        sw = max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3)
        km = kp.numpy()[:, S.P.SEL].astype(np.float64)[2 * sk:]
        a = arms.setdefault(arm, {"runs": [], "bins": [], "mics": set()})
        a["runs"].append(metrics(km, sw, codes[N:, 0]))
        a["bins"].append([phys(km[i:i + 250]) for i in range(0, len(km), 250) if len(km[i:i + 250]) >= 125])
        a["mics"].add(mic)
    out = {}
    for arm in [x for x in ARMS if x in arms]:
        a = arms[arm]
        nb = min(len(b) for b in a["bins"])
        out[arm] = {"label": ARMS[arm], "runs": len(a["runs"]), "mics": sorted(a["mics"]), **mean_rows(a["runs"]),
                    "bins_10s": [mean_rows([b[i] for b in a["bins"]]) for i in range(nb)]}
        print(arm, json.dumps({k: v for k, v in out[arm].items() if k != "bins_10s"}), flush=True)
        print("  bins conf/floor/red/lean95/hroll95", json.dumps([[b["body_conf"], b["below_floor_pct"], b["red_pct"], b["lean_p95"], b["head_roll_p95"]]
                                                                for b in out[arm]["bins_10s"]]), flush=True)
    z = np.load(os.path.join(SC, "work", "phys_real_k55.npz"), allow_pickle=True)
    real = [s for s in z["sets"] if s[0] == "aloe"]
    rows = []
    for s in real:
        k = s[2].astype(np.float64)
        sw = float(s[3]) if np.ndim(s[3]) == 0 else float(np.asarray(s[3]).ravel()[0])
        d = phys(k)
        h = HD.metrics(k, sw)
        d.update(gestures_per_min=h["gestures_per_min"], hands_moving_pct=h["hands_moving_pct"])
        rows.append(d)
    out["REAL_aloe"] = {"label": f"real Aloe held-out B tracks ({len(rows)}, Person 01 frame)", **mean_rows(rows), "cb0_repeat_pct": 58.3,
                        "p10_p90": {k: [r(np.percentile([x[k] for x in rows], 10), 3), r(np.percentile([x[k] for x in rows], 90), 3)] for k in rows[0]}}
    print("REAL", json.dumps(out["REAL_aloe"]), flush=True)
    json.dump(out, open("/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/phys/ab_summary" + ("_seed1" if only1 else "") + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
