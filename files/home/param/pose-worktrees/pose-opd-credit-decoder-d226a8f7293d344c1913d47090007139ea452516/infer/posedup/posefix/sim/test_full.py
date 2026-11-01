#!/usr/bin/env python3
"""posefix GPU check of the STAGED live engine's FULL PREFILL (stage-full/live_ppm_pose.py via posefix/sim/engfull symlinks; inputs from
stage-full/server.py prefill_inputs, exactly as a live Start would build them): call 1086's mic audio as A, the person's voice-clip preset,
graphed steps with NO per-step force (the engine forces lane / Mimi / pose itself). Checks: the forced positions equal the inputs, generation
runs after, ms / frame. Writes <out>/<name>.{npz,json,feed.jsonl} (replay.py format) for rendering.
usage: python test_full.py <ck> <out dir> <person>:<src 1|2>:<seconds> [...]"""
import base64
import json
import os
import sys
import time

E = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/engfull"
sys.path.insert(0, E)
sys.argv_saved = list(sys.argv)
sys.argv = ["server.py"]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import server as S  # noqa: E402   (stage-full server: prefill_inputs, PR, LP = stage-full engine)
sys.argv = sys.argv_saved
LP, PR = S.LP, S.PR
assert os.path.dirname(os.path.abspath(LP.__file__)) == E, LP.__file__
from safetensors.numpy import load_file  # noqa: E402

SPF, CD, OFF, LOOK = LP.SPF, LP.CD, LP.OFF, LP.VS.LOOK24
ck, out, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out, exist_ok=True)
A = np.load("/data/home/amartyavarma/ppm/aloe-posedup-live/sessions/20261025-142453-1086/session.npz")["a"].astype(np.float32)
FANT = json.load(open("/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/runs_dupM.json"))[0]["params"]


def prefix_for(person):
    codes = np.asarray(load_file(f"/data/home/amartyavarma/video/demo-data/prompts/{person}/gt/mimi_local.safetensors")["codes"], np.int64)[:, :125]
    clip = PR.upload_clip(codes)
    clip.update(src="video_person", src_key=f"VIDEO/{person}")
    return PR.build_prefix("", clip, prose=False)[0]


with torch.no_grad():
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    m, ldev, cfgs, info = LP.build_model(ck, devs)
    eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=120.0)
    eng.capture()
    tok = LP.IP.tok()
    res = []
    for spec in specs:
        person, src, sec = spec.split(":")
        pp, pre_f, pinfo = S.prefill_inputs(person, float(sec), int(src), True, True, True, tok)
        N = pinfo["frames"]
        name = f"sf_{person[:2]}_{'start' if pinfo['src'] == 'conversation start' else 'clip'}{sec.replace('.', '')}"
        pre = prefix_for(person)
        P = int(pre["P"])
        params = dict(FANT, pose_light_topk=151)
        eng.start(pre, 777, params, pose_prefill=pp, prefill=pre_f)
        eng.set_params(params)
        nf = N + CD + 375
        rec = {"pose": [], "mimi": [], "tok": [], "b": []}
        wrote, ms = 0, []
        for f in range(nf):
            T = P + f
            need = (f + 1) * SPF + LOOK
            eng.write_a(wrote, A[wrote:need]); wrote = need
            t0 = time.perf_counter()
            dec = eng.step(T)
            tokB, _, _, y = eng.result(T, dec)
            ms.append(1e3 * (time.perf_counter() - t0))
            rec["tok"].append(tokB)
            if y is not None:
                rec["b"].append(y)
            if f >= CD:
                rec["pose"].append(list(eng.last_pose)); rec["mimi"].append(list(eng.last_codes))
        lane = np.asarray(rec["tok"]); pose = np.asarray(rec["pose"]); mimi = np.asarray(rec["mimi"])
        ab = eng.abuf[0, eng.S0:eng.S0 + (N + 4) * SPF].float().cpu().numpy()
        a_silent_ok = bool(np.abs(ab[:N * SPF]).max() == 0.0 and np.abs(ab[N * SPF:]).max() > 0)
        ok_lane = bool((lane[:N + CD] == pre_f["lane"]).all())
        ok_codes = bool((mimi[:N] == pre_f["codes"]).all())
        ok_pose = bool((pose[:N] == pp).all())
        gen = pose[N:, 0]
        r = {"name": name, "person": person, "prefill": pinfo, "P": P, "frames": nf, "lane_forced_ok": ok_lane, "codes_forced_ok": ok_codes,
             "pose_forced_ok": ok_pose, "a_silent_during_prefill_ok": a_silent_ok, "gen_cb0_change": float((gen[1:] != gen[:-1]).mean()), "gen_lane_words": int(((lane[N + CD:] < LP.TE)).sum()),
             "ms_p50": float(np.percentile(ms[5:], 50)), "ms_p95": float(np.percentile(ms[5:], 95)), "pose_prefill_frames": N}
        b = np.concatenate(rec["b"]).astype(np.float32)
        np.savez_compressed(f"{out}/{name}.npz", pose=pose.astype(np.int16), mimi=mimi.astype(np.int16), tok=lane, b=b)
        json.dump(r, open(f"{out}/{name}.json", "w"), indent=1)
        with open(f"{out}/{name}.feed.jsonl", "w") as fo:
            fo.write(json.dumps({"type": "start", "session": name, "model": "posefix-stagefull", "voice": person, "fps": 12.5, "person": person,
                                 "prefill_s": (0.0 if pinfo["src"] == "conversation start" else N * 0.08)}) + "\n")
            for k in range(len(pose)):
                seg = b[k * SPF:(k + 1) * SPF] if len(b) >= (k + 1) * SPF else np.zeros(SPF, np.float32)
                fo.write(json.dumps({"type": "f", "session": name, "f": k, "t": round(k * 0.08, 3), "pose": pose[k].tolist(), "mimi": mimi[k].tolist(),
                                     "pcm_b64": base64.b64encode(np.clip(seg * 32767, -32768, 32767).astype("<i2").tobytes()).decode("ascii")}) + "\n")
            fo.write(json.dumps({"type": "end", "session": name}) + "\n")
        LP.log("TEST " + json.dumps(r))
        res.append(r)
    json.dump(res, open(f"{out}/test_full.json", "w"), indent=1)
    print("ALL OK" if all(x["lane_forced_ok"] and x["codes_forced_ok"] and x["pose_forced_ok"] and x["a_silent_during_prefill_ok"] for x in res) else "FAIL", flush=True)
sys.stdout.flush()
os._exit(0)
