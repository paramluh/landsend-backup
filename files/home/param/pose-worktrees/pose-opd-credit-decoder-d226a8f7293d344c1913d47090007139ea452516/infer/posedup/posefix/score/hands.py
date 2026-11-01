"""posefix 10-26: HAND MOTION / GESTURE metrics on decoded pose keypoints (score.py's kp: [T, 133, 3] px at 25 fps; the 55 rendered joints
= P.SEL: 0-12 body (9 / 10 = left / right wrist), 13-33 left hand (13 = its root), 34-54 right hand (34 = root)). User ask (via main):
"do you have a way of tracking number of gestures or hand motion".
  hand_speed_p50 / p95   per frame transition and hand: mean displacement of that hand's confident-in-both joints (body wrist + 21 hand
                         joints) / the actor's frame-0 horizontal shoulder width (sw); pooled over both hands' valid transitions
  wrist speed            per hand: |d wrist| / sw (body wrist if confident, else the hand root), 0.2 s centred moving average
  hands_moving_pct       100 x share of frames where either wrist speed > MOVE_THR
  gestures_per_min       a gesture = a burst of wrist speed > MOVE_THR lasting >= MIN_S, bursts closer than GAP_S merged (rest >= GAP_S
                         separates); per hand (left / right) and any hand (union of the two moving masks)
Thresholds calibrated on real data (posefix/score/calib_hands.py; held-out Aloe tracks + the demo actors' GT clips)."""
import numpy as np

CONF_T = 0.3
FPS = 25
MOVE_THR = 0.015         # sw / frame (= 0.375 sw / s): calib_hands.py + eye check on 2 rendered runs (hands_sheet.py: 5/5, 6 vs ~5-6 visible)
MIN_S, GAP_S, SMOOTH = 0.3, 0.3, 5
HAND_J = {"left": [9] + list(range(13, 34)), "right": [10] + list(range(34, 55))}
WRIST = {"left": (9, 13), "right": (10, 34)}


def series(k55, sw_a):
    """k55 [T, 55, 3] (x, y, conf) px -> (hand speed [T-1, 2], smoothed wrist speed [T-1, 2]) in sw / frame (NaN = no valid joints)."""
    xy, conf = k55[..., :2], k55[..., 2]
    ok = conf > CONF_T
    hs, ws = [], []
    for h in ("left", "right"):
        J = HAND_J[h]
        d = np.linalg.norm(np.diff(xy[:, J], axis=0), axis=-1) / sw_a
        both = ok[1:, J] & ok[:-1, J]
        n = both.sum(1)
        hs.append(np.where(n > 0, (d * both).sum(1) / np.maximum(n, 1), np.nan))
        a, b = WRIST[h]
        w = np.where(ok[:, a, None], xy[:, a], np.where(ok[:, b, None], xy[:, b], np.nan))
        v = np.linalg.norm(np.diff(w, axis=0), axis=-1) / sw_a
        v0 = np.nan_to_num(v, nan=0.0)
        ker = np.ones(SMOOTH) / SMOOTH
        ws.append(np.convolve(v0, ker, mode="same"))
    return np.stack(hs, 1), np.stack(ws, 1)


def bursts(mask, fps=FPS, min_s=MIN_S, gap_s=GAP_S):
    """bool [T] -> list of (start, end) frame runs: runs of True, gaps < gap_s merged, kept if length >= min_s."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    runs, s, e = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - e - 1 < gap_s * fps:
            e = i
        else:
            runs.append((s, e)); s = e = i
    runs.append((s, e))
    return [(a, b) for a, b in runs if b - a + 1 >= min_s * fps]


def metrics(k55, sw_a, thr=MOVE_THR):
    hs, ws = series(k55, sw_a)
    T = len(ws)
    mins = max(T / FPS / 60.0, 1e-9)
    v = hs[np.isfinite(hs)]
    mv = ws > thr
    anym = mv.any(1)
    g = {h: len(bursts(mv[:, i])) for i, h in enumerate(("left", "right"))}
    ga = len(bursts(anym))
    return {"hand_speed_p50": None if not len(v) else round(float(np.median(v)), 4),
            "hand_speed_p95": None if not len(v) else round(float(np.percentile(v, 95)), 4),
            "hands_moving_pct": round(100 * float(anym.mean()), 1),
            "gestures_per_min": round(ga / mins, 1), "gestures_per_min_left": round(g["left"] / mins, 1),
            "gestures_per_min_right": round(g["right"] / mins, 1), "gestures": ga, "hand_seconds": round(T / FPS, 1)}
