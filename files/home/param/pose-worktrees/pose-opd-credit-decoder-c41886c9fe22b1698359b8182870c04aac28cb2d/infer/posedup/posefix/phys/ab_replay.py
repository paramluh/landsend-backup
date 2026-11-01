#!/usr/bin/env python3
"""posefix PHYS-B 10-26: A/B of pose sampling arms (main GO). Live engine (export-498, graphs), Person 01 voice-clip preset + the 2 s
conversation-start prefill, channel A = 4 mics (the user's 2 calls' processed mic = session.npz 'a', call 1086's mic, held-out conv c2's
partner stem), 2 seeds, 750 frames (60 s). Params = the user's preset (call b62a's session params; the 2 calls use their own) per arm:
  U  = the user's preset (pose CFG 3.2 cb0 + light, pose k34, light k53, T1)
  a  = U with pose_cfg_light 0 (CFG on cb0 only)
  b  = U with pose_cfg_on 0 and pose_light_topk 151
-> posefix/phys/runs_ab/<mic>_<arm>_s<seed>.npz (codes from conversation code frame 0, N). No teacher log-probs.
usage: python ab_replay.py <ck> [frames]"""
import json
import os
import sys
import time

sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/phys")
import replay_attr as RA  # noqa: E402   (server / engine imports, prefix_for)
import numpy as np  # noqa: E402
import torch  # noqa: E402

S, LP = RA.S, RA.LP
SPF, CD, LOOK = RA.SPF, RA.CD, RA.LOOK
OUT = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/phys/runs_ab"
SESS = RA.SESS
PERSON = "01_ab63d745"
ARMS = {"U": {}, "a": {"pose_cfg_light": 0}, "b": {"pose_cfg_on": 0, "pose_light_topk": 151}}


def mics():
    out = {}
    for key, (sid, _) in (("7b5a", RA.CALLS["7b5a"]), ("b62a", RA.CALLS["b62a"])):
        sj = json.load(open(f"{SESS}/{sid}/session.json"))
        out[key] = (np.load(f"{SESS}/{sid}/session.npz")["a"].astype(np.float32), dict(sj["params"]), int(sj["seed"]))
    base = out["b62a"][1]
    out["m1086"] = (np.load("/data/home/amartyavarma/ppm/aloe-posedup-live/sessions/20261025-142453-1086/session.npz")["a"].astype(np.float32), base, 1086)
    out["c2"] = (np.load("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c2.npz")["stem_1"].astype(np.float32), base, 2002)
    return out


@torch.no_grad()
def run(eng, tok, pre, pp, pre_f, N, A, params, seed, nf, out):
    P = int(pre["P"])
    eng.start(pre, seed, params, pose_prefill=pp, prefill=pre_f)
    eng.set_params(params)
    need_tot = (nf + 2) * SPF + LOOK
    if len(A) < need_tot:
        A = np.concatenate([A, np.zeros(need_tot - len(A), np.float32)])
    wrote, codes = 0, []
    for f in range(nf):
        T = P + f
        need = (f + 1) * SPF + LOOK
        eng.write_a(wrote, A[wrote:need]); wrote = need
        dec = eng.step(T)
        eng.result(T, dec)
        if f >= CD:
            codes.append(list(eng.last_pose))
    np.savez_compressed(out, codes=np.asarray(codes, np.int16), N=N, seed=seed, params=json.dumps(params))
    return len(codes)


def main():
    ck = sys.argv[1]
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 750
    os.makedirs(OUT, exist_ok=True)
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    with torch.no_grad():
        m, ldev, cfgs, info = LP.build_model(ck, devs)
        eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=70.0)
        eng.capture()
        tok = LP.IP.tok()
        pre = RA.prefix_for(PERSON)
        pp, pre_f, pinfo = S.prefill_inputs(PERSON, 2.0, 0, True, True, True, tok)
        N = int(pinfo["frames"])
        M = mics()
        print(f"[ab {time.strftime('%H:%M:%S')}] ready: P {pre['P']}, N {N}, mics {list(M)}", flush=True)
        for k in range(2):                                              # seed-major: every (arm x mic) gets seed 1 first
            for arm, patch in ARMS.items():
                for mk, (A, base, s0) in M.items():
                    seed = s0 + 1000 * k
                    out = f"{OUT}/{mk}_{arm}_s{seed}.npz"
                    if os.path.exists(out):
                        continue
                    t0 = time.time()
                    n = run(eng, tok, pre, pp, pre_f, N, A, dict(base, **patch), seed, nf, out)
                    print(f"[ab {time.strftime('%H:%M:%S')}] {mk} {arm} seed {seed}: {n} code frames in {time.time() - t0:.0f} s", flush=True)
    print("[ab] DONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
