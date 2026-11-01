#!/usr/bin/env python3
"""posefix OFFLINE SIMULATOR of the live pose-duplex engine (10-25): the live code path (stage-next/live_ppm_pose.py = the live engine +
POSE PREFILL), the same checkpoint (bf16), the same prompt (the preset's voice clip, no persona = voice_clip_only), the same params and seed
as a recorded session, channel A = the recorded mic audio (session.npz 'a', already the post-level samples the server wrote) fed frame by
frame exactly as server.py does (abuf holds A up to (f + 1) x SPF + LOOK when frame f steps), or a held-out conversation's partner stem.
One process runs a list of runs (JSON) after one load + capture. Per run -> <out>/<name>.npz (pose [F, 32] local ids from code frame 0,
tok, mimi, b audio, cb0 logit stats) + <name>.feed.jsonl (videort's feed format: start / f {f, t, pose, mimi, pcm_b64} / end), and
<name>.json (summary: pose prefill frames, prefill CE of the forced real pose under the model's cb0 = the codec check, cb0 change rate).
usage: python replay.py <ck> <runs.json> <out dir>
run = {name, a: session dir | conv npz, side (conv only, B side K; A = stem_{1-K}), person (avatar / voice preset id, default 01_ab63d745),
       seed (default: the session's), params (overrides on top of the session's params / engine defaults), pose_prefill_s (0), frames}"""
import base64
import json
import os
import sys
import time

LV = "/data/home/amartyavarma/ppm/aloe-posedup-live"
ENG = os.environ.get("POSEFIX_ENGINE_DIR", f"{LV}/posefix/sim/eng")   # symlinks: stage-next/live_ppm_pose.py + the live DEFAULTS.json / silence cache
sys.path.insert(0, LV)
sys.path.insert(0, ENG)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import live_ppm_pose as LP  # noqa: E402
import prompts_ppm as PR  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

assert os.path.dirname(os.path.abspath(LP.__file__)) == os.path.abspath(ENG), LP.__file__
SPF, CD, OFF, LOOK = LP.SPF, LP.CD, LP.OFF, LP.VS.LOOK24
PEOPLE = ["/data/home/amartyavarma/video/demo-data/prompts", "/data/home/amartyavarma/video/dmd-rt/actors/pkg"]


def person_file(person, rel):
    for r in PEOPLE:
        if os.path.isfile(f"{r}/{person}/{rel}"):
            return f"{r}/{person}/{rel}"
    raise FileNotFoundError((person, rel))


def prefix_for(person):
    """= server.add_video_people + Session.prompt for preset <person>, voice_clip_only: the actor's gt/mimi_local [:, :125] voice clip."""
    codes = np.asarray(load_file(person_file(person, "gt/mimi_local.safetensors"))["codes"], np.int64)[:, :125]
    clip = PR.upload_clip(codes)
    clip.update(src="video_person", src_key=f"VIDEO/{person}")
    pre, info = PR.build_prefix("", clip, prose=False)
    return pre, info


def pose_prefill(person, pf):
    """= stage-next server.pose_prefill_codes (pf snapped to whole 0.16 s latents)."""
    if pf <= 0:
        return None
    c = np.asarray(load_file(person_file(person, "gt/pose_tokens.safetensors"))["codes"], np.int64)
    n = int(round(pf / 0.16))
    return c[:, :min(c.shape[1], 2 * n + 2)].T.copy()


@torch.no_grad()
def run_one(eng, r, out):
    name = r["name"]
    person = r.get("person", "01_ab63d745")
    src = r["a"]
    params = {}
    if os.path.isdir(src):
        sj = json.load(open(f"{src}/session.json"))
        A = np.load(f"{src}/session.npz")["a"].astype(np.float32)
        params = {k: v for k, v in sj["params"].items() if k in LP.PARAMS}
        seed = int(r.get("seed", sj["seed"]))
        assert not sj.get("param_log"), "mid-call param changes not replayed"
    else:
        z = np.load(src)
        A = z[f"stem_{1 - int(r.get('side', 0))}"].astype(np.float32)
        seed = int(r.get("seed", 1234))
    params.update(r.get("params", {}))
    if r.get("conv_prefill"):                                         # CONVERSATION OPENING: the conv's own system block (B's voice clip)
        zc = np.load(src)
        K = int(r.get("side", 0))
        pre = {"P": int(zc[f"pre_ids_{K}"].shape[1]), "ids": zc[f"pre_ids_{K}"], "depth": zc[f"pre_depth_{K}"], "present": zc[f"pre_present_{K}"]}
    else:
        pre, info = prefix_for(person)
    P = int(pre["P"])
    pp = pose_prefill(person, float(r.get("pose_prefill_s", 0)))
    FP = None                                                          # FULL PREFILL (pose + words + audio): r["full_prefill"] = {bundle, s}
    if r.get("full_prefill"):
        bz = np.load(r["full_prefill"]["bundle"])
        NF = min(150, 2 * int(round(float(r["full_prefill"]["s"]) / 0.16)) + 2)
        FP = (bz["lane"], bz["audio"][:, :NF].T + OFF[None], bz["pose"][:, :NF].T + OFF[None], NF)
        pp = None
    if r.get("conv_prefill"):                                         # lane / Mimi / pose of B's real opening, frames as the training micro
        NF = min(len(zc[f"laneB_{K}"]) - CD, 2 * int(round(float(r["conv_prefill"]["s"]) / 0.16)) + 2)
        lb, db, pb = zc[f"laneB_{K}"], zc[f"depB_{K}"], zc[f"pdepB_{K}"]
        FP = (lb, db[CD:CD + NF], pb[CD:CD + NF], NF)                  # depB / pdepB: position-indexed (code frame g at index CD + g), global ids
        pp = None
    nf = min(int(r.get("frames", 10 ** 9)), len(A) // SPF - 2, eng.n_fr - 4)
    t0 = time.time()
    eng.start(pre, seed, params, pose_prefill=pp)
    _, bad = eng.set_params(params)
    rec = {"pose": [], "mimi": [], "tok": [], "b": [], "pce": [], "pent": [], "ptop": []}
    wrote = 0
    for f in range(nf):
        T = P + f
        need = (f + 1) * SPF + LOOK
        eng.write_a(wrote, A[wrote:need])
        wrote = need
        frc = None
        if FP is not None and f <= FP[3] + 1:                         # lane frames 0..N+1, Mimi + pose code frames 0..N-1 forced
            g = f - CD
            frc = (int(FP[0][f]), FP[1][g] if 0 <= g < FP[3] else None, FP[2][g] if 0 <= g < FP[3] else None)
        dec = eng.step(T, force=frc)
        tokB, _, _, y = eng.result(T, dec)
        rec["tok"].append(tokB)
        if y is not None:
            rec["b"].append(y)
        cf = f - CD
        if cf >= 0:
            rec["pose"].append(list(eng.last_pose)); rec["mimi"].append(list(eng.last_codes))
            zl = torch.log_softmax(eng.zp0[0].float(), -1)
            ent = float(-(zl.exp() * zl).sum())
            rec["pent"].append(ent)
            rec["ptop"].append(float(zl.max().exp()))
            if pp is not None and cf < len(pp):
                rec["pce"].append(float(-zl[int(pp[cf, 0])]))
            elif FP is not None and cf < FP[3]:
                rec["pce"].append(float(-zl[int(FP[2][cf, 0])]))
    wall = time.time() - t0
    pose = np.asarray(rec["pose"], np.int16)
    b = np.concatenate(rec["b"]) if rec["b"] else np.zeros(0, np.float32)
    np.savez_compressed(f"{out}/{name}.npz", pose=pose, mimi=np.asarray(rec["mimi"], np.int16), tok=np.asarray(rec["tok"]), b=b.astype(np.float32),
                        pent=np.asarray(rec["pent"], np.float32), ptop=np.asarray(rec["ptop"], np.float32), pce=np.asarray(rec["pce"], np.float32))
    with open(f"{out}/{name}.feed.jsonl", "w") as fo:                    # videort replay_feed.py input (= server.feed_msg format)
        fo.write(json.dumps({"type": "start", "session": name, "model": "posefix-sim", "voice": person, "fps": 12.5, "person": person,
                             "prefill_s": float(r.get("video_prefill_s", r.get("pose_prefill_s", 0)))}) + "\n")
        for k in range(len(pose)):
            seg = b[k * SPF:(k + 1) * SPF] if len(b) >= (k + 1) * SPF else np.zeros(SPF, np.float32)
            pcm = np.clip(seg * 32767, -32768, 32767).astype("<i2").tobytes()
            fo.write(json.dumps({"type": "f", "session": name, "f": k, "t": round(k * 0.08, 3), "pose": pose[k].tolist(),
                                 "mimi": rec["mimi"][k], "pcm_b64": base64.b64encode(pcm).decode("ascii")}) + "\n")
        fo.write(json.dumps({"type": "end", "session": name}) + "\n")
    g = int(len(pp)) if pp is not None else (FP[3] if FP is not None else 0)
    p0 = pose[:, 0].astype(int)
    gen = p0[g:]
    summ = {"name": name, "src": src, "person": person, "seed": seed, "P": P, "frames": nf, "pose_frames": int(len(pose)),
            "pose_prefill_frames": g, "pose_prefill_s": g / 12.5, "params": params, "param_problems": bad, "wall_s": round(wall, 1),
            "ms_per_frame": round(1e3 * wall / max(nf, 1), 1),
            "prefill_cb0_ce_real": (float(np.mean(rec["pce"])) if rec["pce"] else None),
            "gen_cb0_change": float((gen[1:] != gen[:-1]).mean()) if len(gen) > 1 else None,
            "gen_cb0_distinct": int(len(np.unique(gen))), "gen_cb0_entropy_mean": float(np.mean(rec["pent"][g:])) if len(rec["pent"]) > g else None,
            "words": " ".join(w["text"] for w in LP.IP.lane_words(np.asarray(rec["tok"])))[:400]}
    json.dump(summ, open(f"{out}/{name}.json", "w"), indent=1)
    LP.log(f"run {name}: {json.dumps({k: summ[k] for k in ('frames', 'pose_prefill_frames', 'prefill_cb0_ce_real', 'gen_cb0_change', 'gen_cb0_distinct', 'ms_per_frame')})}")
    return summ


def main():
    ck, runs, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
    os.makedirs(out, exist_ok=True)
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    m, ldev, cfgs, info = LP.build_model(ck, devs)
    LP.log(f"model {info.get('step')} s {info.get('s')} pose {info.get('pose')} bb {info.get('decp_bb')} alpha {info.get('decp_bb_alpha')}")
    eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=float(os.environ.get("POSEFIX_MAX_S", "600")))
    eng.capture()
    for r in (runs[::-1] if os.environ.get("POSEFIX_REVERSE") else runs):
        if os.path.isfile(f"{out}/{r['name']}.json") and not os.environ.get("POSEFIX_REDO"):
            LP.log(f"skip {r['name']} (done)")
            continue
        try:                                                           # claim: several instances share one run list
            os.close(os.open(f"{out}/{r['name']}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            LP.log(f"skip {r['name']} (claimed)")
            continue
        run_one(eng, r, out)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
