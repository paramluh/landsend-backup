#!/usr/bin/env python3
"""posefix NAME-C step 2: teacher-force Kate's TRAINING row (m1n0 #2937, B = Kate) through the live pose-duplex engine (export-498) and read
B's lane distribution at the frame where she says her name ("... My name is Kate.", lane token ' Kate'). Lane / B Mimi / B pose forced per
step from the trainer's micro (train_side.npz); channel A = the training A waveform (partner's raw FLAC = the trainer's wav). Conditions:
  T0 trainer-exact: EMPTY system block (P = 5, what this row had in training), A = Voxtral(partner audio) from frame 0
  T1 + Kate's clip (P = 134; trainer builder == live builder, step 1), A as T0
  L1 clip + A digital zeros for the first 26 frames (= a live 2 s prefill's a_mute), rest as T1
  L0 no clip + A zeros first 26 frames
usage: python tf_name.py <ck> <out json>"""
import json, os, sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/name/c/eng")
import numpy as np
import torch
import live_ppm_pose as LP
CK, OUT = sys.argv[1], sys.argv[2]
D = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/name/c"
t = np.load(f"{D}/train_side.npz", allow_pickle=True)
tp = np.load(f"{D}/train_side_prefix.npz")
P0, n = int(t["P"]), int(t["n"])
ids, depth, present, pdepth = t["ids"], t["depth"], t["present"], t["pdepth"]
lane = ids[1, P0:]                                   # B lane per conversation frame (+ CD tail)
dep = depth[1, P0:]                                  # B Mimi (global, CD-delayed layout) per frame
pdp = pdepth[P0:]                                    # B pose (global) per frame
wav = t["wav"].astype(np.float32)
tok = LP.IP.tok()
cands = {w: tok.encode(w, add_special_tokens=False) for w in (" Kate", " Lauren", " Rosie", " Katie", " Kat")}
print("token ids", cands, flush=True)
kate = cands[" Kate"][0]
fk = [int(i) for i in np.flatnonzero(lane[:400] == kate)]
print("frames with ' Kate' in the training lane (first 400):", fk, flush=True)
assert fk, "no ' Kate' token in the first 400 lane frames"
F = fk[0]
pre_empty = {"P": P0, "ids": ids[:, :P0], "depth": depth[:, :P0], "present": present[:, :P0]}
pre_clip = {"P": int(tp["P"]), "ids": tp["ids"], "depth": tp["depth"], "present": tp["present"]}
CONDS = [("T0_trainer_noclip_Avox", pre_empty, 0), ("T1_clip_Avox", pre_clip, 0), ("L1_clip_Azero26", pre_clip, 26), ("L0_noclip_Azero26", pre_empty, 26)]
devs = [torch.device("cuda:0"), torch.device("cuda:1")]
res = {"ck": CK, "row": "ALOE/s2handover-none/fdba6c86-caa3-43f0-8249-6e7000d1475b/seg02/c00", "name_frame": F, "token_ids": cands, "conds": {}}
with torch.no_grad():
    m, ldev, cfgs, info = LP.build_model(CK, devs)
    eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=20.0)
    SPF, LOOK, CD = LP.SPF, LP.VS.LOOK24, LP.CD
    for name, pre, nz in CONDS:
        P = int(pre["P"])
        eng.start(pre, 1234, {"pose_cfg_on": 0})
        a = wav[:(F + 4) * SPF + LOOK].copy()
        a[:nz * SPF] = 0.0
        eng.write_a(0, a)
        for f in range(F + 1):
            T = P + f
            frc = (int(lane[f]), dep[f] if dep[f, 0] >= 0 else None, pdp[f] if pdp[f, 0] >= 0 else None)
            eng.step(T, graphs=False, force=frc)
            if f == F:
                z = eng.zt[0].float()                    # cond-row lane logits predicting lane[T] (computed before the forced token)
            eng.result(T, T >= P + CD)
        lp = torch.log_softmax(z, -1)
        isw = torch.arange(z.shape[0], device=z.device) < LP.TE
        lpw = torch.log_softmax(z.masked_fill(~isw, -1e9), -1)    # the word stage (word pieces only)
        top = torch.topk(lp, 10)
        topw = torch.topk(lpw, 10)
        r = {"P": P, "A_zero_frames": nz,
             "top10_raw": [[tok.decode([int(i)]), round(float(p.exp()), 4)] for p, i in zip(top.values, top.indices)],
             "top10_words": [[tok.decode([int(i)]), round(float(p.exp()), 4)] for p, i in zip(topw.values, topw.indices)],
             "p_raw": {w: round(float(lp[c[0]].exp()), 5) for w, c in cands.items()},
             "p_word": {w: round(float(lpw[c[0]].exp()), 5) for w, c in cands.items()}}
        res["conds"][name] = r
        print(f"[name] {name}: {json.dumps(r)}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", flush=True)
os._exit(0)
