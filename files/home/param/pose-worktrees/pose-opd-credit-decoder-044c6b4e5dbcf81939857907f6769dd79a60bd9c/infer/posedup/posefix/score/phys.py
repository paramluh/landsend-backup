"""posefix 10-26: PER-FRAME PHYSICALITY score for decoded pose (user: "very unrealistic poses ... the llm shouldnt be able to generate non
physical motion"). Keypoints exactly as score.py builds them (S.decode + S.to_keypoints with the person's frame 0; the 55 rendered joints
= P.SEL: 0-12 body (COCO: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips), 13-33 left hand, 34-54 right
hand (21 each: root, then thumb / index / middle / ring / pinky x 4)); 25 fps; confident joints only (conf > 0.3); every length / speed in
units of the actor's frame-0 horizontal shoulder width (sw).
TESTS (frame score > tau = flagged; each score is the worst bone / joint / finger relative to its own REAL band):
  bone_len   bone length outside the real per-bone band [p0.5, p99.5] (54 bones: body 12, wrist->hand-root 2, hand 40); score = max excess
  bone_jump  frame-to-frame change of a bone's length / its real p99.5 change
  vel        joint speed / its real per-joint p99.5;  acc = |x[t+1] - 2 x[t] + x[t-1]| / its real per-joint p99.5
  angle      max of: finger zig-zag (consecutive turning angles of one finger with opposite signs: min |theta|) and finger turn |theta|, each /
             its real per-finger p99.5; elbow fold (interior angle below the real p0.5 with both arm bones >= 0.5 x their real median
             length, i.e. not foreshortened); wrist bend (forearm vs hand root->middle base) / real p99.5
  (self-crossing limbs: NOT scored: crossed arms / clasped or crossed fingers are physical and indistinguishable in 2D)
tau per test = the pooled REAL frame-score quantile 0.998 (real: held-out Aloe tracks in Person 01's frame + the 20 demo actors' GT
clips, own frame 0) -> every test flags 0.2 % of real frames by construction, 'any' <= 1 %. Calibration: phys_calib.py -> out/phys_calib.json.
API: frames(k55, sw) -> (scores [T, 5], flags [T, 5]);  phys_stream(codes, person, start_token) -> dict (+ per-frame arrays under '_frames')."""
import json
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
CONF_T = float(os.environ.get("PHYS_CONF", "0.3"))   # 0.3 = confident joints (default); 0.2 = 'as rendered' (raster conf_min 0.2: the
# skeleton the video model sees draws every joint above 0.2, red at 0.2-0.4), calibrated separately -> phys_calib_c020.json
CALIB = os.path.join(H, "out", "phys_calib.json" if CONF_T == 0.3 else f"phys_calib_c{int(round(CONF_T * 100)):03d}.json")
TESTS = ("bone_len", "bone_jump", "vel", "acc", "angle")
BODY_BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (1, 2), (3, 4), (0, 1), (0, 2), (5, 11), (6, 12), (11, 12)]
LINK_BONES = [(9, 13), (10, 34)]
FINGERS = [[r] + [r + 1 + 4 * f + i for i in range(4)] for r in (13, 34) for f in range(5)]          # 10 chains of 5 joints
HAND_BONES = [(c[i], c[i + 1]) for c in FINGERS for i in range(4)]
BONES = BODY_BONES + LINK_BONES + HAND_BONES
BI = np.asarray(BONES)
ARMS = [(5, 7, 9), (6, 8, 10)]                       # shoulder, elbow, wrist (left, right)
HANDDIR = [(7, 9, 13, 22), (8, 10, 34, 43)]          # elbow, wrist, hand root, middle-finger base (root + 9)
_CAL = {}


def calib():
    if "c" not in _CAL:
        _CAL["c"] = json.load(open(CALIB))
    return _CAL["c"]


def _ang(u, v):
    """signed angle from u to v (deg), [..., 2] vectors."""
    cr = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    dt = (u * v).sum(-1)
    return np.degrees(np.arctan2(cr, dt))


def features(k55, sw):
    """k55 [T, 55, 3] (x, y, conf) px -> dict of raw per-frame features (NaN where not confident)."""
    xy = k55[..., :2] / sw
    ok = k55[..., 2] > CONF_T
    T = len(xy)
    xyn = np.where(ok[..., None], xy, np.nan)
    L = np.linalg.norm(xyn[:, BI[:, 0]] - xyn[:, BI[:, 1]], axis=-1)                                 # [T, nb]
    J = np.full_like(L, np.nan)
    J[1:] = np.abs(L[1:] - L[:-1])
    V = np.full((T, 55), np.nan)
    V[1:] = np.linalg.norm(xyn[1:] - xyn[:-1], axis=-1)
    A = np.full((T, 55), np.nan)
    A[1:-1] = np.linalg.norm(xyn[2:] - 2 * xyn[1:-1] + xyn[:-2], axis=-1)
    # fingers: turning angles at the 3 inner joints of each chain
    ch = np.asarray(FINGERS)                                                                            # [10, 5]
    seg = xyn[:, ch[:, 1:]] - xyn[:, ch[:, :-1]]                                                        # [T, 10, 4, 2]
    th = _ang(seg[:, :, :-1], seg[:, :, 1:])                                                            # [T, 10, 3]
    opp = np.sign(th[:, :, 1:]) * np.sign(th[:, :, :-1]) < 0
    zig = np.where(opp, np.minimum(np.abs(th[:, :, 1:]), np.abs(th[:, :, :-1])), 0.0)
    zig = np.where(np.isfinite(th[:, :, 1:]) & np.isfinite(th[:, :, :-1]), zig, np.nan).max(-1) if T else np.zeros((0, 10))
    turn = np.nanmax(np.abs(th), -1) if T else np.zeros((0, 10))
    # elbows: interior angle + both bone lengths; wrist bend
    elb, ua, fa, wb = [], [], [], []
    for (s, e, w) in ARMS:
        u, v = xyn[:, s] - xyn[:, e], xyn[:, w] - xyn[:, e]
        elb.append(np.abs(_ang(u, v)))
        ua.append(np.linalg.norm(u, axis=-1)); fa.append(np.linalg.norm(v, axis=-1))
    for (e, w, r, m) in HANDDIR:
        wb.append(np.abs(_ang(xyn[:, w] - xyn[:, e], xyn[:, m] - xyn[:, r])))
    return {"L": L, "J": J, "V": V, "A": A, "zig": zig, "turn": turn, "elb": np.stack(elb, 1), "ua": np.stack(ua, 1),
            "fa": np.stack(fa, 1), "wb": np.stack(wb, 1)}


def _nanmax0(x):
    with np.errstate(all="ignore"):
        m = np.nanmax(np.where(np.isfinite(x), x, -np.inf), -1)
    return np.where(np.isfinite(m), m, 0.0)


def scores(F, c):
    """features + calibration bands -> per-frame test scores [T, 5] (score > 1 = beyond the real per-item band; tau applied after)."""
    lo, hi = np.asarray(c["len_lo"]), np.asarray(c["len_hi"])
    L = F["L"]
    with np.errstate(all="ignore"):
        e = np.maximum(np.maximum(L / hi - 1.0, lo / np.maximum(L, 1e-6) - 1.0), 0.0)
        s_len = _nanmax0(np.where(np.isfinite(L), e, np.nan))
        s_jump = _nanmax0(F["J"] / np.asarray(c["jump_hi"]))
        s_vel = _nanmax0(F["V"] / np.asarray(c["vel_hi"]))
        s_acc = _nanmax0(F["A"] / np.asarray(c["acc_hi"]))
        zig = F["zig"] / np.asarray(c["zig_hi"])
        turn = F["turn"] / np.asarray(c["turn_hi"])
        longb = (F["ua"] >= 0.5 * np.asarray(c["ua_med"])) & (F["fa"] >= 0.5 * np.asarray(c["fa_med"]))
        elb = np.where(longb, np.asarray(c["elb_lo"]) / np.maximum(F["elb"], 0.5), np.nan)
        wb = F["wb"] / np.asarray(c["wb_hi"])
        s_ang = np.maximum.reduce([_nanmax0(zig), _nanmax0(turn), _nanmax0(elb), _nanmax0(wb)])
    return np.stack([s_len, s_jump, s_vel, s_acc, s_ang], 1)


def frames(k55, sw, c=None):
    """-> (scores [T, 5], flags [T, 5] bool) with the calibrated tau per test."""
    c = c or calib()
    s = scores(features(k55, sw), c)
    return s, s > np.asarray([c["tau"][t] for t in TESTS])[None]


def summarize(flags, name=None):
    T = len(flags)
    out = {"frames": int(T), "fail_any_pct": round(100 * float(flags.any(1).mean()), 2) if T else None}
    out.update({f"fail_{t}_pct": round(100 * float(flags[:, i].mean()), 2) if T else None for i, t in enumerate(TESTS)})
    if name:
        out["name"] = name
    return out


def phys_stream(codes, person, start_token=0, device="cpu"):
    """codes [F, 32] local pose ids (from conversation code frame 0) -> summary dict over the measured frames (tokens >= start_token;
    decode normalised to token 0, as the avatar) + '_frames': {'scores': [T, 5], 'flags': [T, 5]} (numpy, not JSON)."""
    import score as S
    ck, sk, _ = S.prep_codes(S._fix_codes(codes), start_token)
    kp, kp0 = S.to_keypoints(S.decode(ck, device), S.frame0_of(person))
    k0 = kp0.numpy()
    sw = max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3)
    lo = 2 * sk
    k55 = kp.numpy()[:, S.P.SEL].astype(np.float64)
    s, f = frames(k55, sw)
    out = summarize(f[lo:])
    out["_frames"] = {"scores": s[lo:], "flags": f[lo:]}
    return out
