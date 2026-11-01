#!/usr/bin/env python3
"""posefix 10-26: GPU check of the stage-ring fix: during a FULL PREFILL the word ring (frequency / repetition penalty) must hold exactly
the FED lane tokens. Staged engine via posefix/sim/engring (stage-ring symlinks), prefill inputs from stage-ring/server.py, call-1086 mic as A,
graphed steps without per-step force. After every forced frame: engine ring state == a reference LiveRing pushed with the forced lane so far.
usage: python test_ring.py <ck> <person>:<src>:<seconds> [...]"""
import json
import os
import sys

E = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/engring"
sys.path.insert(0, E)
saved, sys.argv = list(sys.argv), ["server.py"]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import server as S  # noqa: E402
sys.argv = saved
LP, PR = S.LP, S.PR
assert os.path.dirname(os.path.abspath(LP.__file__)) == E, LP.__file__
from safetensors.numpy import load_file  # noqa: E402

SPF, CD, LOOK = LP.SPF, LP.CD, LP.VS.LOOK24
A = np.load("/data/home/amartyavarma/ppm/aloe-posedup-live/sessions/20261025-142453-1086/session.npz")["a"].astype(np.float32)
FANT = json.load(open("/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/runs_dupM.json"))[0]["params"]
with torch.no_grad():
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    m, ldev, cfgs, info = LP.build_model(sys.argv[1], devs)
    eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=60.0)
    eng.capture()
    tok = LP.IP.tok()
    allok = True
    for spec in sys.argv[2:]:
        person, src, sec = spec.split(":")
        pp, pre_f, pinfo = S.prefill_inputs(person, float(sec), int(src), True, True, True, tok)
        codes = np.asarray(load_file(f"/data/home/amartyavarma/video/demo-data/prompts/{person}/gt/mimi_local.safetensors")["codes"], np.int64)[:, :125]
        clip = PR.upload_clip(codes); clip.update(src="video_person", src_key=f"VIDEO/{person}")
        pre = PR.build_prefix("", clip, prose=False)[0]
        P = int(pre["P"])
        params = dict(FANT, pose_light_topk=151)
        eng.start(pre, 4242, params, pose_prefill=pp, prefill=pre_f)
        eng.set_params(params)
        lane = pre_f["lane"]
        ref = LP.LiveRing(eng.ring.V, eng.ring.TE, eng.dl, eng.ring.excl)
        bad, words, wrote = [], 0, 0
        for f in range(len(lane) + 40):
            T = P + f
            need = (f + 1) * SPF + LOOK
            eng.write_a(wrote, A[wrote:need]); wrote = need
            dec = eng.step(T)
            eng.result(T, dec)
            if f < len(lane):
                ref.push(torch.full((1,), int(lane[f]), dtype=torch.long, device=eng.dl))
                same = all(bool(torch.equal(a, b)) for a, b in zip(eng.ring.state(), ref.state()))
                if not same:
                    bad.append(f)
        words = int(ref.ptr.item())
        ok = not bad
        allok &= ok
        print(json.dumps({"spec": spec, "prefill": pinfo, "forced_lane_frames": len(lane), "word_pieces_in_ring": words,
                          "ring_matches_fed_tokens_every_forced_frame": ok, "first_bad": bad[:5]}), flush=True)
    print("ALL OK" if allok else "FAIL", flush=True)
sys.stdout.flush()
os._exit(0)
