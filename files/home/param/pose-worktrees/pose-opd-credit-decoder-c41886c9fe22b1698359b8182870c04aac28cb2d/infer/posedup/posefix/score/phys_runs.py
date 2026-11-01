#!/usr/bin/env python3
"""posefix 10-26: phys.py on generated pose: the user's two live calls (genweb runs 20261026-205615-7b5a, -205651-b62a), recent monoplex
playground runs and live sessions; % frames failing per test (generated frames only, after the prefill), drift in 10 s bins, and contact
sheets of the user calls' worst frames from render/out/<rid>_sxs.mp4 (avatar | skeleton). -> out/phys_runs.json + work/phys_worst_<rid>.png"""
import glob
import json
import os
import subprocess
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import score as S  # noqa: E402
import phys as PH  # noqa: E402

LIVE = "/data/home/amartyavarma/ppm/aloe-posedup-live"
RUNS, ROUT = f"{LIVE}/genweb/runs", f"{LIVE}/genweb/render/out"
FFD = "/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static"
USER_CALLS = ["20261026-205615-7b5a", "20261026-205651-b62a"]
BIN = 250                                                            # 10 s of 25 fps pose frames


def run_inputs(rd):
    summ = json.load(open(os.path.join(rd, "summary.json")))
    codes = np.load(os.path.join(rd, "tokens.npz"))["pose"]
    N = int(summ.get("pose_prefill_frames") or summ.get("prefill_frames") or 0)
    return codes, summ.get("person") or S.DEFAULT_PERSON, N, summ


def drift(flags):
    return [round(100 * float(flags[i:i + BIN].any(1).mean()), 2) for i in range(0, len(flags), BIN) if len(flags[i:i + BIN]) >= BIN // 2]


def score_one(name, codes, person, N):
    r = PH.phys_stream(codes, person, N)
    fr = r.pop("_frames")
    r.update(name=name, person=person, prefill_tokens=N, drift_any_pct_per_10s=drift(fr["flags"]))
    return r, fr


def worst_sheet(rid, fr, N, codes_len, k=20):
    """the k worst measured frames (max score / tau) from the run's _sxs.mp4 -> work/phys_worst_<rid>.png; returns their times."""
    src = os.path.join(ROUT, f"{rid}_sxs.mp4")
    if not os.path.exists(src):
        return None
    tau = np.asarray([PH.calib()["tau"][t] for t in PH.TESTS])
    rel = (fr["scores"] / tau[None]).max(1)
    order = [int(i) for i in np.argsort(-rel)]
    pick = []
    for i in order:                                                  # spread: >= 0.5 s apart
        if all(abs(i - j) >= 12 for j in pick):
            pick.append(i)
        if len(pick) >= k:
            break
    pick.sort()
    dur = float(subprocess.run([f"{FFD}/ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                               capture_output=True, text=True).stdout.strip() or 0)
    off = 2 * N if dur > codes_len * 0.08 - 1.0 else 0               # the render starts at token 0 (prefill included) or at the gen
    sel = "+".join(f"eq(n,{off + i})" for i in pick)
    out = os.path.join(H, "work", f"phys_worst_{rid}.png")
    subprocess.run([f"{FFD}/ffmpeg", "-y", "-v", "error", "-i", src, "-vf", f"select='{sel}',scale=320:-2,tile=5x4", "-frames:v", "1", out],
                   check=True)
    tests = [PH.TESTS[int(np.argmax(fr["scores"][i] / tau))] for i in pick]
    return {"png": out, "frames": [{"t_s": round((off + i) / 25, 2), "rel": round(float(rel[i]), 2), "test": t} for i, t in zip(pick, tests)]}


def main():
    res = {"user_calls": [], "monoplex": [], "sessions": []}
    for rid in USER_CALLS:
        codes, person, N, _ = run_inputs(os.path.join(RUNS, rid))
        r, fr = score_one(rid, codes, person, N)
        r["worst"] = worst_sheet(rid, fr, N, len(codes))
        np.savez_compressed(os.path.join(H, "work", f"phys_frames_{rid}.npz"), **fr)
        res["user_calls"].append(r)
        print("CALL", json.dumps({k: v for k, v in r.items() if k != "worst"}), flush=True)
    mono = []
    for rd in sorted(glob.glob(f"{RUNS}/2026*"), key=os.path.getmtime, reverse=True):
        try:
            codes, person, N, summ = run_inputs(rd)
        except (OSError, ValueError, KeyError):
            continue
        if str(summ.get("model", "")).startswith("monoplex") and len(codes) - N >= 200:
            mono.append((os.path.basename(rd), codes, person, N))
        if len(mono) >= 10:
            break
    for name, codes, person, N in mono:
        r, _ = score_one(name, codes, person, N)
        res["monoplex"].append(r)
        print("MONO", json.dumps(r), flush=True)
    for sd in sorted(glob.glob(f"{LIVE}/sessions/2026102*"), key=os.path.getmtime, reverse=True)[:12]:
        try:
            z = np.load(os.path.join(sd, "session.npz"))
            sj = json.load(open(os.path.join(sd, "session.json")))
        except (OSError, ValueError):
            continue
        if "pose" not in z.files or len(z["pose"]) < 300:
            continue
        N = int(sj.get("pose_prefill_frames") or 0)
        r, _ = score_one(os.path.basename(sd), z["pose"], sj.get("person") or S.DEFAULT_PERSON, N)
        res["sessions"].append(r)
        print("SESS", json.dumps(r), flush=True)
    for k in ("monoplex", "sessions"):
        if res[k]:
            res[f"{k}_mean"] = {f"fail_{t}_pct": round(float(np.mean([r[f"fail_{t}_pct"] for r in res[k]])), 2) for t in ("any",) + PH.TESTS}
    json.dump(res, open(os.path.join(H, "out", "phys_runs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
