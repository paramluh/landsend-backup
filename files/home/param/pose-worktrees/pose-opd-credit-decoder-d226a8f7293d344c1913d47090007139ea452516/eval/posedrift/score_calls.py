#!/usr/bin/env python3
"""posedrift 10-26: attribute low pose confidence + posture drift to sampling settings from EXISTING recorded calls (no GPU, no replays).
Every recorded pose stream (live sessions: sessions/*, stage-lb/*/sessions/*; genweb runs: duplex playground/live + monoplex) is decoded
exactly as posefix/score (score.decode + to_keypoints with the person's frame 0 = the avatar), generated frames only (token >= prefill),
then per 25 fps frame: joint conf (55 rendered joints), body conf (0-12), share of joints <= 0.2 (below the raster floor), 0.2-0.3 (red),
posture (posefix phys_posture.geo: torso lean, torso shear, shoulder roll, head roll) + head pitch (score.head_angles pitch / pitch_nose),
phys 'any' kinematic fail (phys.py, conf 0.3). Aggregated per call + per 10 s bin (time = seconds since call start, prefill included).
Real reference = posefix work/phys_real_k55.npz 'aloe' held-out tracks (+ actor GT clips), same metrics.
usage: score_calls.py inventory            -> out/inventory.json
       score_calls.py run SHARD NSHARDS NPROC -> out/calls.<shard>.jsonl
       score_calls.py real NPROC          -> out/real.jsonl"""
import glob
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/data/home/amartyavarma/.cache/huggingface")
os.environ.setdefault("OMP_NUM_THREADS", "1")
H = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.dirname(H)
SC = f"{LIVE}/posefix/score"
sys.path.insert(0, SC)
OUT = f"{H}/out"
BIN = 250                                                            # 10 s of 25 fps frames
POSE_KEYS = ("pose_cfg_on", "pose_cfg_s", "pose_cfg_cb0", "pose_cfg_light", "pose_cfg_speaking_only", "pose_temp", "pose_topk",
             "pose_light_temp", "pose_light_topk", "pose_rep", "pose_rep_light", "pose_rep_win", "pose_rep_cb0_speaking",
             "pose_rep_cb0_silent", "pose_rep_light_speaking", "pose_rep_light_silent", "cfg_on", "cfg_s", "cb0_topk", "light_topk",
             "min_p", "text_temp")


def _j(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def inventory():
    items, seen = [], {}
    sess = sorted(glob.glob(f"{LIVE}/sessions/2026*")) + sorted(glob.glob(f"{LIVE}/stage-lb/*/sessions/2026*"))
    for d in sess:
        sj = _j(f"{d}/session.json")
        if sj is None or not os.path.exists(f"{d}/session.npz"):
            continue
        p = sj.get("params") or {}
        cfg = sj.get("cfg") or {}
        it = {"id": os.path.basename(d), "src": "live_session", "dir": d, "npz": f"{d}/session.npz", "key": "pose",
              "prefill": int(sj.get("pose_prefill_frames") or 0), "person": sj.get("person"), "step": sj.get("step"),
              "ck": os.path.basename(str(sj.get("ck") or "")), "model": "duplex-498" if sj.get("step") == 498 else f"duplex-{sj.get('step')}",
              "client": cfg.get("client"), "label": sj.get("label"), "inst": d.split("/stage-lb/")[1].split("/")[0] if "/stage-lb/" in d else "main",
              "params": {k: p.get(k) for k in POSE_KEYS}, "param_log": sj.get("param_log") or [],
              "cfg_guided_frames": sj.get("cfg_guided_frames"), "cfg_frames_total": sj.get("cfg_frames_total"),
              "missing": sj.get("pose_missing_frames")}
        seen[it["id"]] = len(items)
        items.append(it)
    for d in sorted(glob.glob(f"{LIVE}/genweb/runs/2026*")):
        s = _j(f"{d}/summary.json")
        if s is None or not os.path.exists(f"{d}/tokens.npz"):
            continue
        rid = os.path.basename(d)
        if s.get("session_dir") and os.path.basename(s["session_dir"]) in seen:
            items[seen[os.path.basename(s["session_dir"])]]["genweb_run"] = rid     # same call: keep the session copy
            continue
        p = s.get("params") or {}
        m = str(s.get("model") or "")
        req = _j(f"{d}/request.json") or {}
        items.append({"id": rid, "src": "genweb", "dir": d, "npz": f"{d}/tokens.npz", "key": "pose",
                      "prefill": int(s.get("pose_prefill_frames") or s.get("prefill_frames") or 0), "person": s.get("person"),
                      "step": s.get("step"), "ck": os.path.basename(str(s.get("ck") or "")),
                      "model": "monoplex-357" if m.startswith("monoplex") else ("duplex-498" if "498" in m else m[:40]),
                      "client": req.get("client") or ("genweb-" + str(s.get("dup_mode") or "mono")), "label": m,
                      "dup_mode": s.get("dup_mode"), "params": {k: p.get(k) for k in POSE_KEYS}, "param_log": s.get("param_log") or [],
                      "hidden": bool(req.get("hidden"))})
    json.dump(items, open(f"{OUT}/inventory.json", "w"), indent=1)
    print(len(items), "items")
    return items


# ------------------------------------------------------------------------------------------------ metrics
def _geo(k):
    import numpy as np
    from phys_posture import geo
    import score as S
    r, l, s, h, c = geo(k)
    ha = S.head_angles(k[..., :2])
    return {"sh_roll": r, "lean": l, "shear": s, "head_roll": h, "pitch": ha["pitch"], "pitch_nose": ha["pitch_nose"]}


def frame_metrics(k55, sw):
    """k55 [T, 55, 3] -> dict of per-frame arrays."""
    import numpy as np
    import phys as PH
    c = k55[..., 2]
    g = _geo(k55)
    f = PH.frames(k55, sw)[1].any(1)
    return {"conf": c.mean(1), "body_conf": c[:, :13].mean(1), "hand_conf": c[:, 13:].mean(1),
            "below_floor": (c <= 0.2).mean(1), "red": ((c > 0.2) & (c <= 0.3)).mean(1), "phys_any": f.astype(float), **g}


AGG_MEAN = ("conf", "body_conf", "hand_conf", "below_floor", "red", "phys_any")
AGG_MED = ("lean", "shear", "sh_roll", "head_roll", "pitch", "pitch_nose")


def summarise(fm, t0_frames=0):
    """per-call means + per-10 s bins + OLS slopes per minute (time since call start) + early (<=20 s) / late (>=40 s) means."""
    import numpy as np
    T = len(fm["conf"])
    t = (t0_frames + np.arange(T)) / 25.0
    out = {"frames": int(T), "gen_s": round(T / 25, 1), "end_s": round(float(t[-1]) if T else 0, 1)}
    for k in AGG_MEAN:
        out[k] = round(float(np.nanmean(fm[k])), 4)
    for k in AGG_MED:
        out[k] = round(float(np.nanmedian(fm[k])), 4)
    bins = []
    for i in range(0, T, BIN):
        sl = slice(i, i + BIN)
        if T - i < BIN // 2:
            break
        b = {"t0_s": round(float(t[i]), 1)}
        for k in AGG_MEAN:
            b[k] = round(float(np.nanmean(fm[k][sl])), 4)
        for k in AGG_MED:
            b[k] = round(float(np.nanmedian(fm[k][sl])), 3)
        bins.append(b)
    out["bins"] = bins
    sl_ = {}
    if T >= 25 * 20:
        tm = t / 60.0
        for k in ("conf", "body_conf", "below_floor", "lean", "shear", "pitch", "pitch_nose", "head_roll"):
            y = fm[k]
            ok = np.isfinite(y)
            if ok.sum() > 100:
                sl_[k] = round(float(np.polyfit(tm[ok], y[ok], 1)[0]), 5)
    out["slope_per_min"] = sl_
    early, late = t <= 20, t >= 40
    if late.sum() >= 250 and early.sum() >= 125:
        out["late_minus_early"] = {k: round(float(np.nanmean(fm[k][late]) - np.nanmean(fm[k][early])), 4)
                                   for k in ("conf", "body_conf", "below_floor", "lean", "shear", "pitch", "pitch_nose", "head_roll")}
    return out


_INIT = {}


def _init():
    import torch
    torch.set_num_threads(1)
    import score as S
    _INIT["S"] = S


def score_item(it):
    import numpy as np
    import score as S
    try:
        z = np.load(it["npz"])
        if it["key"] not in z.files:
            return {**it, "error": "no pose key"}
        codes = S._fix_codes(z[it["key"]])
        person = it.get("person") or S.DEFAULT_PERSON
        if not os.path.isdir(os.path.join(S.PROMPTS, person)):
            person = S.DEFAULT_PERSON
        ck, sk, dropped = S.prep_codes(codes, it["prefill"])
        if len(ck) - sk < 25:
            return {**it, "error": f"short {len(ck) - sk}"}
        kp, kp0 = S.to_keypoints(S.decode(ck), S.frame0_of(person))
        k0 = kp0.numpy()
        sw = max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3)
        k55 = kp.numpy()[:, S.P.SEL].astype(np.float64)[2 * sk:]
        fm = frame_metrics(k55, sw)
        return {**it, "person_used": person, "dropped": dropped, **summarise(fm, 2 * sk)}
    except Exception as e:  # noqa: BLE001
        return {**it, "error": repr(e)[:300]}


def run(shard, nshards, nproc):
    import multiprocessing as mp
    items = json.load(open(f"{OUT}/inventory.json"))
    mine = items[shard::nshards]
    with mp.get_context("fork").Pool(nproc, initializer=_init) as pool, open(f"{OUT}/calls.{shard}.jsonl", "w") as f:
        for r in pool.imap_unordered(score_item, mine):
            f.write(json.dumps(r) + "\n")
            f.flush()
    print("done", shard, len(mine), flush=True)


def _real_one(a):
    import numpy as np
    kind, name, k, sw, max_s = a
    k = k.astype(np.float64)[: int(max_s * 25)]
    return {"kind": kind, "name": str(name), **summarise(frame_metrics(k, float(sw)))}


def real(nproc):
    import multiprocessing as mp
    import numpy as np
    z = np.load(f"{SC}/work/phys_real_k55.npz", allow_pickle=True)
    args = [(s[0], s[1], s[2], s[3], 600) for s in z["sets"]]
    with mp.get_context("fork").Pool(nproc, initializer=_init) as pool, open(f"{OUT}/real.jsonl", "w") as f:
        for r in pool.imap_unordered(_real_one, args):
            f.write(json.dumps(r) + "\n")
    print("real done", len(args), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inventory":
        inventory()
    elif cmd == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "real":
        real(int(sys.argv[2]))
