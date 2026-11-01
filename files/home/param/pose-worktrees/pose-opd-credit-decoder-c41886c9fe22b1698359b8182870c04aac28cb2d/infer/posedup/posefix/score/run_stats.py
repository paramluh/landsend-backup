#!/usr/bin/env python3
"""posefix 10-26: PER-RUN POSE STATS for the playground (user via main: cb0 repeat %, out-of-frame %, keypoint confidence on every output).
Thin wrapper on score.py (same decode, same metric definitions) for one genweb run dir:
  tokens.npz 'pose' [n, 32] local ids from conversation code frame 0 (the forced prefill frames included);
  summary.json 'pose_prefill_frames' (else 'prefill_frames', else 0) = N: measured = generated tokens only (token >= N);
  person = summary.json 'person' (else request.json prefill.person / prompt.voice, else 01_ab63d745); decode normalised to the stream's frame
  0 (token 0) with that person's frame-0 root + shoulder width (= the avatar), no filters (the renderer applies none).
Definitions (generated frames only):
  cb0_repeat_pct    100 x share of consecutive generated tokens whose pose cb0 equals the previous one (= 100 x (1 - score cb0_change))
  out_of_frame_pct  100 x mean over 25 fps frames of (confident joints outside the 480x256 canvas / confident joints); confident = conf > 0.3
  conf              mean keypoint confidence over the 55 rendered joints (score conf_all); lowconf_pct = 100 x share of joint-frames <= 0.3
REF = real data scored the same way (posefix/score/out gt_aloe.json [no filters] and gt_actors.json): mean (p10-p90).
CPU only (~1-2 s for a 30 s run after the first call loads the tokenizer); call run_stats(run_dir) per run, or keep one process for many.
usage: python run_stats.py RUN_DIR [RUN_DIR ...]   -> one JSON line per run"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import score as S  # noqa: E402
import hands as HD  # noqa: E402   (10-26 hand motion / gestures)

REF = {
    "aloe_heldout": {"label": "real Aloe held-out tracks (24, 10 min each)", "cb0_repeat_pct": 58.3, "cb0_repeat_p10_p90": [41.9, 74.6],
                     "out_of_frame_pct": 21.1, "out_of_frame_p10_p90": [0.9, 43.1], "conf": 0.43, "conf_p10_p90": [0.28, 0.52],
                     "note": "real tracks placed in Person 01's frame: ~21 % of confident joints fall outside from crop mismatch alone",
                     # hands (posefix/score/calib_hands.py, first 4 min of each track; out/hands_calib.json)
                     "gestures_per_min": 10.9, "gestures_per_min_p10_p90": [6.6, 15.1], "hands_moving_pct": 21.4,
                     "hand_speed_p50": 0.0065, "hand_speed_p95": 0.073},
    "person01_clip": {"label": "Person 01's own GT clip (12 s)", "cb0_repeat_pct": 33.6, "out_of_frame_pct": 6.2, "conf": 0.51,
                      "gestures_per_min": 30.1, "hands_moving_pct": 64.5, "hand_speed_p50": 0.0306, "hand_speed_p95": 0.1675},
    "actors_clips": {"label": "the 20 demo actors' GT clips (12 s each)", "gestures_per_min": 19.1, "gestures_per_min_p10_p90": [10.0, 30.1],
                     "hands_moving_pct": 41.7, "hand_speed_p50": 0.0129, "hand_speed_p95": 0.1195},
}


ACTOR_REFS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "actor_refs.json")   # actor_refs.py (per-actor GT)


def actor_ref(person):
    """the person's own GT-clip reference (actor_refs.json; None if absent). Re-read per call: the file may be regenerated."""
    try:
        return json.load(open(ACTOR_REFS_FILE)).get(person)
    except (OSError, ValueError):
        return None


def _prefill_frames(run_dir, summ):
    for k in ("pose_prefill_frames", "prefill_frames"):
        if summ.get(k) is not None:
            return int(summ[k])
    return 0


def _person(run_dir, summ):
    if summ.get("person"):
        return summ["person"]
    try:
        req = json.load(open(os.path.join(run_dir, "request.json")))
        return (req.get("prefill") or {}).get("person") or (req.get("prompt") or {}).get("voice") or S.DEFAULT_PERSON
    except (OSError, ValueError):
        return S.DEFAULT_PERSON


def run_stats(run_dir, device="cpu"):
    t0 = time.time()
    run_dir = os.path.normpath(run_dir)
    try:
        summ = json.load(open(os.path.join(run_dir, "summary.json")))
    except (OSError, ValueError):
        summ = {}
    codes = S._fix_codes(np.load(os.path.join(run_dir, "tokens.npz"))["pose"])
    N = _prefill_frames(run_dir, summ)
    person = _person(run_dir, summ)
    if person not in os.listdir(S.PROMPTS):
        person = S.DEFAULT_PERSON
    ck, start_kept, _ = S.prep_codes(codes, N)
    assert len(ck) > start_kept + 2, ("stream too short", len(ck), start_kept)
    kp, kp0 = S.to_keypoints(S.decode(ck, device), S.frame0_of(person))
    lo = 2 * start_kept
    w = S.metrics(kp, kp0, ck, lo, kp.shape[0])                    # = score_stream's 'whole' (filters none)
    k0 = kp0.numpy()
    hm = HD.metrics(kp.numpy()[:, S.P.SEL].astype(np.float64)[lo:], max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3))
    r = {"measured_tokens": int(len(ck) - start_kept)}
    out = {"run": os.path.basename(run_dir), "person": person, "prefill_tokens": N, "measured_tokens": r["measured_tokens"],
           "cb0_repeat_pct": None if w.get("cb0_change") is None else round(100 * (1 - w["cb0_change"]), 1),
           "out_of_frame_pct": None if w.get("out_of_frame") is None else round(100 * w["out_of_frame"], 1),
           "conf": w.get("conf_all"), "lowconf_pct": None if w.get("lowconf_frac") is None else round(100 * w["lowconf_frac"], 1),
           **hm,
           "ref": REF, "ref_person": actor_ref(person), "scorer": "posefix/score/run_stats.py (score.py definitions)",
           "secs": round(time.time() - t0, 2)}
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(run_stats(d)), flush=True)
