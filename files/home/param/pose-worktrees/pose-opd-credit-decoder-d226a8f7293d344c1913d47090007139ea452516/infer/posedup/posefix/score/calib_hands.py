"""posefix 10-26: calibrate hands.py thresholds on REAL pose: held-out Aloe tracks (run_baselines.aloe_tracks, person 01 frame, first 3000
tokens = 4 min each) + the demo actors' own GT clips (own frame 0). Prints wrist-speed percentiles and, per candidate threshold, the real
% moving and gestures / min, -> out/hands_calib.json."""
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import score as S  # noqa: E402
import hands as HD  # noqa: E402
import run_baselines as RB  # noqa: E402


def k55_of(codes, person):
    codes, _, _ = S.prep_codes(codes)
    dec = S.decode(codes, "cpu")
    kp, kp0 = S.to_keypoints(dec, S.frame0_of(person))
    k = kp.numpy()[:, S.P.SEL].astype(np.float64)
    k0 = kp0.numpy()
    return k, max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3)


def main():
    sets = {"aloe": [], "actors": []}
    for conv, side, path in RB.aloe_tracks()[:24]:
        c, _ = S.load_codes(path)
        sets["aloe"].append((f"{conv}:{side}", *k55_of(c[:3000], RB.PERSON)))
    for d in sorted(os.listdir(S.PROMPTS)):
        p = os.path.join(S.PROMPTS, d, "gt", "pose_tokens.safetensors")
        if os.path.exists(p):
            c, _ = S.load_codes(p)
            sets["actors"].append((d, *k55_of(c, d)))
    out = {}
    for nm, rows in sets.items():
        ws = np.concatenate([HD.series(k, sw)[1].ravel() for _, k, sw in rows])
        hs = np.concatenate([HD.series(k, sw)[0].ravel() for _, k, sw in rows])
        hs = hs[np.isfinite(hs)]
        pct = {p: round(float(np.percentile(ws, p)), 4) for p in (10, 25, 50, 75, 90, 95)}
        cand = {}
        for thr in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
            ms = [HD.metrics(k, sw, thr) for _, k, sw in rows]
            cand[thr] = {"moving_pct": round(float(np.mean([m["hands_moving_pct"] for m in ms])), 1),
                         "gpm": round(float(np.mean([m["gestures_per_min"] for m in ms])), 1),
                         "gpm_p10_p90": [round(float(x), 1) for x in np.percentile([m["gestures_per_min"] for m in ms], [10, 90])]}
        out[nm] = {"n": len(rows), "wrist_speed_pct": pct, "hand_speed_p50": round(float(np.median(hs)), 4),
                   "hand_speed_p95": round(float(np.percentile(hs, 95)), 4), "by_threshold": cand}
        if nm == "actors":
            p01 = [HD.metrics(k, sw) for n_, k, sw in rows if n_ == "01_ab63d745"]
            out[nm]["person01_default_thr"] = p01[0] if p01 else None
        print(nm, json.dumps(out[nm]), flush=True)
    json.dump(out, open(os.path.join(H, "out", "hands_calib.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
