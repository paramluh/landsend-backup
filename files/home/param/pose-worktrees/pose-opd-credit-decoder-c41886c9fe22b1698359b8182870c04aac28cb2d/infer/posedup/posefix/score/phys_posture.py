"""posefix 10-26: posture drift per 10 s bin (user call b62a / 7b5a vs real Aloe tracks): shoulder-line roll |deg|, torso lean |deg|
(shoulder mid -> hip mid vs vertical), torso shear (hip width / shoulder width), head roll |deg| (eye line), body conf, share of frames
with |shoulder roll| > real p99 or |lean| > real p99. -> out/phys_posture.json"""
import json, os, sys
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import numpy as np
from phys_drift import k55_run


def geo(k):
    x = k[..., :2]
    sh = x[:, 6] - x[:, 5]
    roll = np.degrees(np.arctan2(sh[:, 1], sh[:, 0]))
    roll = np.abs(((roll + 90) % 180) - 90)
    ms, mh = (x[:, 5] + x[:, 6]) / 2, (x[:, 11] + x[:, 12]) / 2
    v = mh - ms
    lean = np.abs(np.degrees(np.arctan2(v[:, 0], v[:, 1])))
    shear = np.linalg.norm(x[:, 12] - x[:, 11], axis=-1) / np.maximum(np.linalg.norm(sh, axis=-1), 1e-3)
    ey = x[:, 2] - x[:, 1]
    hroll = np.degrees(np.arctan2(ey[:, 1], ey[:, 0])); hroll = np.abs(((hroll + 90) % 180) - 90)
    return roll, lean, shear, hroll, k[:, :13, 2].mean(1)


def bins(k, lim, B=250):
    r, l, s, h, c = geo(k)
    out = []
    for i in range(0, len(r), B):
        if len(r[i:i + B]) < B // 2:
            break
        sl = slice(i, i + B)
        out.append({"t0_s": i / 25, "sh_roll": round(float(np.median(r[sl])), 1), "lean": round(float(np.median(l[sl])), 1),
                    "shear": round(float(np.median(s[sl])), 2), "head_roll": round(float(np.median(h[sl])), 1),
                    "body_conf": round(float(c[sl].mean()), 3),
                    "bad_posture_pct": round(100 * float(((r[sl] > lim[0]) | (l[sl] > lim[1])).mean()), 1)})
    return out


def main():
    z = np.load(os.path.join(H, "work", "phys_real_k55.npz"), allow_pickle=True)
    real = [s for s in z["sets"] if s[0] == "aloe"]
    rr = [geo(k.astype(np.float64)) for _, _, k, _ in real]
    R, L = np.concatenate([g[0] for g in rr]), np.concatenate([g[1] for g in rr])
    lim = (float(np.percentile(R, 99)), float(np.percentile(L, 99)))
    res = {"real_limits_p99": {"sh_roll": round(lim[0], 1), "lean": round(lim[1], 1)},
           "real_all": {"sh_roll_p50": round(float(np.median(R)), 1), "lean_p50": round(float(np.median(L)), 1),
                        "bad_posture_pct": round(100 * float(((R > lim[0]) | (L > lim[1])).mean()), 2)}}
    for rid in ("20261026-205651-b62a", "20261026-205615-7b5a"):
        k, sw = k55_run(rid)
        res[rid] = bins(k, lim)
    res["real_aloe_first3"] = [bins(k.astype(np.float64), lim)[:8] for _, _, k, _ in real[:3]]
    json.dump(res, open(os.path.join(H, "out", "phys_posture.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "real_aloe_first3"}), flush=True)


if __name__ == "__main__":
    main()
