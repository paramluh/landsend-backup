#!/usr/bin/env python3
"""posefix POSE-QUALITY SCORER: posev3 token streams -> keypoints exactly as the live avatar renders them -> health metrics.

Decode = video/code/pose (PoseTokenizer.decode with TF32 off, denormalised with a person's frame-0 root + shoulder width via
pose.global_state / kinematic.offsets_to_positions, 480x256 px, SEL = the 55 joints), then optionally the live renderer's own
filters, imported from video/dmd-rt/rt/live.py (LivePose._orient then LivePose._align, the same order and ramps as LivePose._next).

Inputs (any mix): .npy / .npz ([F, 32] or [32, F] local ids, or global ids with +cb*2048 offsets; npz key pose / codes / pdepB_K),
a live session dir (session.npz key 'pose'; feed.jsonl fallback), a feed.jsonl ('f' lines with 'pose' [32]),
a gt/pose_tokens.safetensors (key 'codes' [32, F]). Frames with any id < 0 (live: the first 2) are dropped.

usage: score.py IN [IN ...] [--person 01_ab63d745 | --person own] [--filters none|align|orient|align+orient]
                [--start-token N] [--max-tokens N] [--window-s 30] [--device cpu|cuda] [--out OUT.json]
--start-token: decode from token 0 (normalisation is the stream's frame 0) but measure only tokens >= N (offline samples: 125).
--person own: a gt/pose_tokens.safetensors input uses its own prompt dir's frame0_pose.

Metric definitions (frames = 25 fps pose frames, 2 per token; px in the 480x256 (HxW) canvas; sw_a = the ACTOR's frame-0 horizontal
shoulder width in px; 'confident' = conf > 0.3):
  conf_body / conf_hand / conf_all   mean keypoint confidence over joints 0-12 / 13-54 / all 55 (conf_all = pose_stats 'conf')
  lowconf_frac                       fraction of joint-frames with conf < 0.3; conf_flicker = per joint-frame rate of crossing 0.3
  speed_p50 / p95, jumps_per_s       per frame transition: mean px displacement of joints confident in both frames / sw_a;
                                     jumps = transitions with speed > 0.08
  ps_speed_p50 / p95, ps_jumps_per_s, ps_still   EXACT video/dmd-rt/rt/pose_stats.py definitions (all 55 joints, normalised
                                     (y/H, x/W) coords, per-frame horizontal width), on the final (filtered) keypoints
  centre_own_* / centre_actor_*      |shoulder centre - (stream frame-0 | actor frame-0) shoulder centre| / sw_a (p50, p95, end)
  width_own_* / width_actor_*        horizontal shoulder width / (stream frame-0 | actor frame-0) width (p05, p50, p95, end)
                                     (actor versions = live.py LivePose._align framing stats); end = mean of the last 2 s
  out_of_frame                       per frame: confident joints outside [0, W-1] x [0, H-1] / confident joints, mean (= live);
                                     _body / _hand pooled over joints 0-12 / 13-54; _bottom_share = share of those leaving at y > H-1
  head pitch / yaw / roll            pitch = (eye-mid y - ear-mid y) / eye dist (>0 = looking down), pitch_nose = (nose y -
                                     eye-mid y) / eye dist, yaw = (nose x - eye-mid x) / eye dist, roll = eye-line angle (deg);
                                     slow component = trailing 2 s mean; *_p50 = median of slow, *_drift = slow(end) - slow(first 2 s),
                                     *_sd_slow = std of slow (held offsets wandering), ref = the actor frame-0 value; frames with
                                     eye distance < 0.04 sw_a are skipped (head_valid_frac = share kept)
  width_collapse_frac                frames whose horizontal shoulder width < 0.5 x the actor's (collapsed / side-on body)
  cb0_change, cb0_distinct, cb0_distinct_per30s, light_change   token stats on the measured tokens (light = cb1-31 mean)
  prop_*                             medians over frames: eye dist, ear dist, upper arm (5-7, 6-8), forearm (7-9, 8-10), each / the
                                     frame's Euclidean shoulder distance |5-6|; (needs both joints of a segment confident)
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

VIDEO_CODE = "/data/home/amartyavarma/video/code"
RT = "/data/home/amartyavarma/video/dmd-rt/rt"
PROMPTS = "/data/home/amartyavarma/video/demo-data/prompts"
sys.path.insert(0, VIDEO_CODE)
sys.path.insert(0, RT)
import pose as P  # noqa: E402
from pose.kinematic import offsets_to_positions  # noqa: E402

DEFAULT_PERSON = "01_ab63d745"
FPS = 25
CONF_T = 0.3
_TOK = {}


# ------------------------------------------------------------------------------------------------ inputs
def _fix_codes(a):
    """-> int64 [F, 32] local ids, -1 kept (dropped later)."""
    a = np.asarray(a).astype(np.int64)
    assert a.ndim == 2 and 32 in a.shape, a.shape
    if a.shape[1] != 32:  # [32, F] (a 32 x 32 array is taken as [F, 32])
        a = a.T
    if (a >= 2048).any():  # global ids (+cb * 2048)
        off = np.arange(32, dtype=np.int64)[None] * 2048
        a = np.where(a >= 0, a - off, -1)
    assert ((a >= -1) & (a < 2048)).all(), (a.min(), a.max())
    return a


def _from_feed(path):
    rows = {}
    for line in open(path):
        m = json.loads(line)
        if m.get("type", "f") == "f" and "pose" in m:
            rows[int(m["f"])] = m["pose"]
    assert rows, path
    n = max(rows) + 1
    a = np.full((n, 32), -1, np.int64)
    for f, p in rows.items():
        a[f] = p
    return a


def load_codes(path, npz_key=None):
    """-> (codes int64 [F, 32] local ids with -1 frames still in, info dict)."""
    info = {"src": path}
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "session.npz")):
            z = np.load(os.path.join(path, "session.npz"))
            if "pose" in z.files:
                info["kind"] = "session.npz"
                return _fix_codes(z["pose"]), info
        info["kind"] = "feed.jsonl"
        return _fix_codes(_from_feed(os.path.join(path, "feed.jsonl"))), info
    if path.endswith(".jsonl"):
        info["kind"] = "feed.jsonl"
        return _fix_codes(_from_feed(path)), info
    if path.endswith(".safetensors"):
        from safetensors.numpy import load_file
        z = load_file(path)
        info["kind"] = "safetensors"
        return _fix_codes(z[npz_key or "codes"]), info
    if path.endswith(".npz"):
        z = np.load(path)
        key = npz_key or next(k for k in ("pose", "codes", "pdepB_0") if k in z.files)
        info["kind"], info["key"] = "npz", key
        return _fix_codes(z[key]), info
    if path.endswith(".npy"):
        info["kind"] = "npy"
        return _fix_codes(np.load(path)), info
    raise ValueError(f"unknown input {path}")


def frame0_of(person):
    from safetensors.torch import load_file
    return load_file(os.path.join(PROMPTS, person, "llm_inputs.safetensors"))["frame0_pose"]


# ------------------------------------------------------------------------------------------------ decode (= pose.decode / LivePose)
def tokenizer(device):
    if device not in _TOK:
        from huggingface_hub import snapshot_download
        from pose.pose_tokenizer import PoseTokenizer
        _TOK[device] = PoseTokenizer.from_pretrained(snapshot_download(P.TOKENIZER[0], revision=P.TOKENIZER[1]), device=device)
    return _TOK[device]


@torch.no_grad()
def decode(codes_f32, device="cpu"):
    """codes int [F, 32] (no -1) -> dec [2F, 165] float (= pose.decode: TF32 off on CUDA)."""
    tok = tokenizer(device)
    c = torch.as_tensor(codes_f32).long().T  # [32, F]
    if device.startswith("cuda"):
        return P.decode(tok, c)
    return tok.decode([x[None] for x in c.unbind(0)]).float()[0].cpu()


@torch.no_grad()
def to_keypoints(dec, frame0_pose):
    """dec [T, 165] -> kp [T, 133, 3] (x, y, conf) px, exactly as LivePose._next / pose.tokens_to_latents."""
    yxc = torch.as_tensor(frame0_pose).float().numpy()
    kp0 = torch.from_numpy(np.ascontiguousarray(yxc[:, [1, 0, 2]]))
    gs = P.global_state(kp0.numpy())
    T = dec.shape[0]
    f = dec.view(T, 55, 3)
    g = gs.expand(T, 3)
    pos = offsets_to_positions(g[:, :2], f[..., :2] * g[:, None, 2:3])
    kp = torch.zeros(T, 133, 3)
    kp[:, P.SEL, :2] = torch.stack([pos[..., 1] * P.W, pos[..., 0] * P.H], -1)
    kp[:, P.SEL, 2] = f[..., 2]
    return kp, kp0


def apply_filters(kp, frame0_pose, filters):
    """LivePose's own _orient then _align (imported from rt/live.py, unbound on a CPU-only instance; same ramps / state)."""
    if filters == "none":
        return kp
    from live import LivePose
    lp = LivePose.__new__(LivePose)
    lp.P = P
    lp.start(frame0_pose, orient="orient" in filters, align="align" in filters)
    kp = kp.clone()
    frames = range(kp.shape[0])
    kp = lp._orient(kp, frames)
    kp = lp._align(kp, frames)
    return kp


# ------------------------------------------------------------------------------------------------ metrics
def _r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def _slow(x, n=50):
    """trailing mean over n frames (causal), = LivePose 'shapes[-50:]'; NaN frames are skipped (NaN if none in the window)."""
    ok = np.isfinite(x)
    c = np.cumsum(np.insert(np.where(ok, x, 0.0), 0, 0.0))
    k = np.cumsum(np.insert(ok.astype(np.float64), 0, 0.0))
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(0, i - n)
    cnt = k[i] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[i] - c[lo]) / np.maximum(cnt, 1), np.nan)


def head_angles(xy):
    """xy [T, 55, 2] (x, y) px -> dict of per-frame proxies."""
    nose, le, re, lear, rear = (xy[:, i] for i in range(5))
    em, am = (le + re) / 2, (lear + rear) / 2
    ed = np.maximum(np.linalg.norm(le - re, axis=-1), 1e-6)
    d = le - re
    return {"pitch": (em[:, 1] - am[:, 1]) / ed, "pitch_nose": (nose[:, 1] - em[:, 1]) / ed,
            "yaw": (nose[:, 0] - em[:, 0]) / ed, "roll": np.degrees(np.arctan2(d[:, 1], d[:, 0]))}


def proportions(xy, conf):
    """median segment lengths / Euclidean shoulder distance over frames where both ends (and both shoulders) are confident."""
    ok = conf > CONF_T
    shd = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=-1)
    oks = ok[:, 5] & ok[:, 6] & (shd > 1e-3)
    out = {}
    segs = {"eye": [(1, 2)], "ear": [(3, 4)], "upper_arm": [(5, 7), (6, 8)], "forearm": [(7, 9), (8, 10)]}
    for name, pairs in segs.items():
        vals = []
        for a, b in pairs:
            m = oks & ok[:, a] & ok[:, b]
            vals.append(np.linalg.norm(xy[m, a] - xy[m, b], axis=-1) / shd[m])
        v = np.concatenate(vals)
        out[f"prop_{name}"] = _r(np.median(v)) if len(v) else None
        out[f"prop_{name}_n"] = int(len(v))
    out["prop_sh_over_hw"] = _r(np.median(shd[oks] / np.maximum(np.abs(xy[oks, 5, 0] - xy[oks, 6, 0]), 1e-3))) if oks.any() else None
    return out


def frame0_ref(kp0):
    """the actor's own frame-0 values (kp0 [133, 3] (x, y, conf) px)."""
    k = kp0.numpy()[P.SEL][None]
    xy, conf = k[..., :2].astype(np.float64), k[..., 2]
    h = {a: _r(v[0]) for a, v in head_angles(xy).items()}
    pr = proportions(xy, np.maximum(conf, CONF_T + 1e-3))  # frame 0 is the reference: take every joint
    pr = {k2: v for k2, v in pr.items() if not k2.endswith("_n")}
    return {"head": h, **pr, "conf_body": _r(conf[0, :13].mean()), "conf_hand": _r(conf[0, 13:].mean())}


def metrics(kp, kp0, codes, lo_f, hi_f):
    """kp [T, 133, 3] final keypoints (whole decoded stream), kp0 actor frame 0; measure frames [lo_f, hi_f) and the tokens
    codes [lo_f // 2, hi_f // 2)."""
    k = kp.numpy()[:, P.SEL].astype(np.float64)
    xy_all, conf_all = k[..., :2], k[..., 2]
    xy, conf = xy_all[lo_f:hi_f], conf_all[lo_f:hi_f]
    T = len(xy)
    k0 = kp0.numpy()
    c_actor = (k0[5, :2] + k0[6, :2]) / 2
    sw_a = max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3)
    out = {"frames": T, "seconds": round(T / FPS, 2)}
    # confidence
    okc = conf > CONF_T
    out.update(conf_all=_r(conf.mean()), conf_body=_r(conf[:, :13].mean()), conf_hand=_r(conf[:, 13:].mean()),
               lowconf_frac=_r(1 - okc.mean()), lowconf_body=_r(1 - okc[:, :13].mean()), lowconf_hand=_r(1 - okc[:, 13:].mean()),
               conf_flicker=_r((okc[1:] != okc[:-1]).mean()) if T > 1 else None)
    # speed (confident joints, actor frame-0 sw)
    if T > 1:
        d = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / sw_a
        both = okc[1:] & okc[:-1]
        sp = np.where(both.any(1), (d * both).sum(1) / np.maximum(both.sum(1), 1), np.nan)
        sp = sp[np.isfinite(sp)]
        out.update(speed_p50=_r(np.median(sp), 5) if len(sp) else None, speed_p95=_r(np.percentile(sp, 95), 5) if len(sp) else None,
                   jumps_per_s=_r((sp > 0.08).sum() / (T / FPS), 3))
        # pose_stats definitions (all joints, normalised coords, per-frame horizontal width)
        pn = np.stack([xy[..., 1] / P.H, xy[..., 0] / P.W], -1)
        width = np.maximum(np.abs(pn[:, 5, 1] - pn[:, 6, 1]), 1e-6)
        vel = np.linalg.norm(np.diff(pn, axis=0), axis=-1).mean(-1) / width[1:]
        out.update(ps_speed_p50=_r(np.median(vel), 5), ps_speed_p95=_r(np.percentile(vel, 95), 5),
                   ps_jumps_per_s=_r((vel > 0.08).sum() / (T / FPS), 3), ps_still=_r((vel < 0.002).mean(), 3))
    # framing
    c = (xy[:, 5] + xy[:, 6]) / 2
    sw = np.abs(xy[:, 5, 0] - xy[:, 6, 0])
    c_own, sw_own = (xy_all[0, 5] + xy_all[0, 6]) / 2, max(abs(float(xy_all[0, 5, 0] - xy_all[0, 6, 0])), 1e-3)
    tail = slice(max(0, T - 2 * FPS), T)
    for nm, ref_c in (("own", c_own), ("actor", c_actor)):
        dc = np.linalg.norm(c - ref_c, axis=-1) / sw_a
        out.update({f"centre_{nm}_p50": _r(np.median(dc), 3), f"centre_{nm}_p95": _r(np.percentile(dc, 95), 3),
                    f"centre_{nm}_end": _r(dc[tail].mean(), 3)})
        dv = (c - ref_c) / sw_a
        out.update({f"centre_{nm}_dx_p50": _r(np.median(dv[:, 0]), 3), f"centre_{nm}_dy_p50": _r(np.median(dv[:, 1]), 3)})
    for nm, ref_w in (("own", sw_own), ("actor", sw_a)):
        wr = sw / ref_w
        out.update({f"width_{nm}_p05": _r(np.percentile(wr, 5), 3), f"width_{nm}_p50": _r(np.median(wr), 3),
                    f"width_{nm}_p95": _r(np.percentile(wr, 95), 3), f"width_{nm}_end": _r(wr[tail].mean(), 3)})
    oof = ((xy[..., 0] < 0) | (xy[..., 0] > P.W - 1) | (xy[..., 1] < 0) | (xy[..., 1] > P.H - 1)) & okc
    out["out_of_frame"] = _r((oof.sum(1) / np.maximum(okc.sum(1), 1)).mean())
    out["out_of_frame_any"] = _r(oof.any(1).mean(), 3)  # fraction of frames with >= 1 confident joint outside
    out["out_of_frame_body"] = _r(oof[:, :13].sum() / max(okc[:, :13].sum(), 1))  # joints 0-12 / 13-54 (joint-frame pooled)
    out["out_of_frame_hand"] = _r(oof[:, 13:].sum() / max(okc[:, 13:].sum(), 1))
    out["out_of_frame_bottom_share"] = _r((oof & (xy[..., 1] > P.H - 1)).sum() / max(oof.sum(), 1), 3) if oof.any() else None
    # head angles (slow = trailing 2 s mean, computed on the whole decoded stream so the window edge has history). Frames whose eye
    # distance < 0.04 actor sw (~8 px; eyes collapsed / both clamped onto a frame corner by align) are excluded (NaN).
    ha = head_angles(xy_all)
    ed_all = np.linalg.norm(xy_all[:, 1] - xy_all[:, 2], axis=-1)
    hv = ed_all >= 0.04 * sw_a
    out["head_valid_frac"] = _r(hv[lo_f:hi_f].mean(), 3)
    out["width_collapse_frac"] = _r((sw / sw_a < 0.5).mean(), 3)  # frames with horizontal shoulder width < half the actor's
    for a, v in ha.items():
        v = np.where(hv, v, np.nan)
        s = _slow(v)[lo_f:hi_f]
        f = np.isfinite(s)
        if f.sum() < 2:
            out.update({f"{a}_p50": None, f"{a}_sd_slow": None, f"{a}_drift": None, f"{a}_fast_sd": None})
            continue
        st, en = s[:min(T, 2 * FPS)], s[tail]
        out.update({f"{a}_p50": _r(np.nanmedian(s), 3), f"{a}_sd_slow": _r(np.nanstd(s), 3),
                    f"{a}_drift": _r(np.nanmean(en) - np.nanmean(st), 3) if np.isfinite(en).any() and np.isfinite(st).any() else None,
                    f"{a}_fast_sd": _r(np.nanstd(v[lo_f:hi_f] - s), 3)})
    # body proportions
    out.update(proportions(xy, conf))
    # tokens
    lo_t, hi_t = lo_f // 2, hi_f // 2
    ct = codes[lo_t:hi_t]
    if len(ct) > 1:
        ch = ct[1:] != ct[:-1]
        n30 = int(30 * 12.5)
        dist30 = [len(np.unique(ct[i:i + n30, 0])) for i in range(0, len(ct), n30) if len(ct[i:i + n30]) >= n30 // 3]
        out.update(cb0_change=_r(ch[:, 0].mean(), 3), light_change=_r(ch[:, 1:].mean(), 3), cb0_distinct=int(len(np.unique(ct[:, 0]))),
                   cb0_distinct_per30s=_r(np.mean(dist30), 1) if dist30 else None)
    return out


def prep_codes(codes_raw, start_token=0, max_tokens=None):
    """drop -1 frames; -> (codes [F, 32], kept tokens before start_token, dropped count)."""
    keep = (codes_raw >= 0).all(1)
    start_kept = int(keep[:start_token].sum())
    codes = codes_raw[keep]
    if max_tokens:
        codes = codes[:start_kept + max_tokens]
    return codes, start_kept, int((~keep).sum())


def score_stream(codes_raw, frame0_pose, filters="none", start_token=0, max_tokens=None, window_s=30.0, device="cpu", name=None, info=None,
                 dec=None):
    """codes_raw [F, 32] (may hold -1 frames) -> result dict. dec: a precomputed decode(prep_codes(...)[0]) to reuse."""
    t0 = time.time()
    codes, start_kept, n_drop = prep_codes(codes_raw, start_token, max_tokens)
    assert len(codes) > start_kept + 2, ("stream too short", len(codes), start_kept)
    if dec is None:
        dec = decode(codes, device)
    assert dec.shape[0] == 2 * len(codes), (dec.shape, len(codes))
    t1 = time.time()
    kp, kp0 = to_keypoints(dec, frame0_pose)
    kp = apply_filters(kp, frame0_pose, filters)
    T = kp.shape[0]
    lo_f = 2 * start_kept
    res = {"name": name, "info": info or {}, "filters": filters, "tokens_total": int(len(codes)), "tokens_dropped_neg": n_drop,
           "start_token": start_kept, "measured_tokens": int(len(codes) - start_kept), "decode_s": round(t1 - t0, 2)}
    res["whole"] = metrics(kp, kp0, codes, lo_f, T)
    W = int(round(window_s * FPS))
    res["windows"] = []
    for s in range(lo_f, T, W):
        e = min(T, s + W)
        if e - s < W // 3:
            break
        m = metrics(kp, kp0, codes, s, e)
        m["t0_s"], m["t1_s"] = round((s - lo_f) / FPS, 1), round((e - lo_f) / FPS, 1)
        res["windows"].append(m)
    res["ref_actor_frame0"] = frame0_ref(kp0)
    res["score_s"] = round(time.time() - t0, 2)
    return res


def person_for(path, person):
    if person == "own":
        d = os.path.dirname(os.path.dirname(os.path.abspath(path)))  # prompts/<id>/gt/pose_tokens.safetensors
        return os.path.basename(d)
    return person


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--person", default=DEFAULT_PERSON)
    ap.add_argument("--filters", default="none", choices=["none", "align", "orient", "align+orient"])
    ap.add_argument("--start-token", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--npz-key", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    results = []
    for path in a.inputs:
        codes, info = load_codes(path, a.npz_key)
        person = person_for(path, a.person)
        info["person"] = person
        r = score_stream(codes, frame0_of(person), a.filters, a.start_token, a.max_tokens, a.window_s, a.device,
                         name=os.path.basename(os.path.normpath(path)), info=info)
        w = r["whole"]
        print(f"== {r['name']} [{a.filters}, {person}] {w['seconds']}s conf body {w['conf_body']} hand {w['conf_hand']} "
              f"speed p50 {w.get('speed_p50')} jumps/s {w.get('jumps_per_s')} | ps speed p50 {w.get('ps_speed_p50')} "
              f"ps jumps/s {w.get('ps_jumps_per_s')} | centre actor p50 {w['centre_actor_p50']} width actor p50 {w['width_actor_p50']} "
              f"oof {w['out_of_frame']} | cb0 change {w.get('cb0_change')} distinct/30s {w.get('cb0_distinct_per30s')} "
              f"({r['decode_s']}s decode)", flush=True)
        results.append(r)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(results if len(results) > 1 else results[0], open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
