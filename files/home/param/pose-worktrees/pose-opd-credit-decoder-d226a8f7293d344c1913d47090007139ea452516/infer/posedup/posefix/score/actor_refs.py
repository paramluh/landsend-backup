#!/usr/bin/env python3
"""posefix 10-26: PER-ACTOR references for the playground's pose stats (user: each prefill actor's OWN real rate as the reference).
Each demo actor's GT clip (video/demo-data/prompts/<pid>/gt/pose_tokens.safetensors, ~12 s), decoded with the actor's OWN frame 0, scored
with exactly run_stats.py's definitions (score.metrics + hands.metrics, all tokens measured, no filters) -> out/actor_refs.json
{pid: {cb0_repeat_pct, gestures_per_min, gestures_per_min_left/right, gestures, hands_moving_pct, hand_speed_p50/p95, out_of_frame_pct,
conf, lowconf_pct, seconds}}. run_stats.run_stats() attaches the run's own person's entry as 'ref_person'.
Caveat: 12 s clips -> gestures_per_min moves in steps of 5 (one gesture = 5 / min)."""
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import score as S  # noqa: E402
import hands as HD  # noqa: E402


def score_codes(codes, person, start_token=0):
    ck, sk, _ = S.prep_codes(S._fix_codes(codes), start_token)
    kp, kp0 = S.to_keypoints(S.decode(ck), S.frame0_of(person))
    lo = 2 * sk
    w = S.metrics(kp, kp0, ck, lo, kp.shape[0])
    k0 = kp0.numpy()
    hm = HD.metrics(kp.numpy()[:, S.P.SEL].astype(np.float64)[lo:], max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3))
    return {"cb0_repeat_pct": None if w.get("cb0_change") is None else round(100 * (1 - w["cb0_change"]), 1),
            "out_of_frame_pct": None if w.get("out_of_frame") is None else round(100 * w["out_of_frame"], 1),
            "conf": w.get("conf_all"), "lowconf_pct": None if w.get("lowconf_frac") is None else round(100 * w["lowconf_frac"], 1),
            "seconds": round((kp.shape[0] - lo) / 25, 1), **hm}


def main():
    out = {}
    for pid in sorted(os.listdir(S.PROMPTS)):
        p = os.path.join(S.PROMPTS, pid, "gt", "pose_tokens.safetensors")
        if not os.path.exists(p):
            continue
        codes, _ = S.load_codes(p)
        out[pid] = score_codes(codes, pid)
        print(pid, json.dumps(out[pid]), flush=True)
    meta = {"_about": "per-actor GT-clip references (posefix/score/actor_refs.py; run_stats.py definitions; own frame 0; all tokens)"}
    json.dump(dict(meta, **out), open(os.path.join(H, "out", "actor_refs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
