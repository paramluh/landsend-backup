#!/usr/bin/env python3
"""posefix 10-26: ADAPTIVE POSE REP-PEN for the live pose-duplex engine (user: rep-pen only while speaking, optional) — fixed-seed proof.
Old = the deployed aloe-posedup-live/live_ppm_pose.py, new = stage-adapt (pose_rep_{cb0,light}_{speaking,silent}, = fast_ppm semantics).
Same model, seed, prompt (held-out conv c1's system block) and A audio (its partner stem); free generation with CUDA graphs:
  identical cases (new knobs off / equal to the static pose rep-pens) -> lane / Mimi / pose byte-equal; one active case must differ.
usage: python test_adapt_live.py <ck> <out json> [frames]"""
import importlib.util
import json
import os
import sys

E = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/engadapt"
sys.path.insert(0, E)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import live_ppm_pose as LN  # noqa: E402
assert os.path.dirname(os.path.realpath(LN.__file__)).endswith("stage-adapt"), LN.__file__
_sp = importlib.util.spec_from_file_location("lp_old", "/data/home/amartyavarma/ppm/aloe-posedup-live/live_ppm_pose.py")
LO = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(LO)

CK, OUT = sys.argv[1], sys.argv[2]
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 150
SPF, CD, LOOK = LN.SPF, LN.CD, LN.VS.LOOK24
CONV = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c1.npz"
ST = {"pose_rep": 1.3, "pose_rep_light": 1.2, "pose_rep_win": 64}
NEWEQ = {"pose_rep_cb0_speaking": 1.3, "pose_rep_cb0_silent": 1.3, "pose_rep_light_speaking": 1.2, "pose_rep_light_silent": 1.2}
CASES = [("defaults", {}, {}),
         ("static_adapt_off", dict(ST, adaptive_rep_on=0), dict(ST, adaptive_rep_on=0)),
         ("static_adapt_off_newknobs_set", dict(ST, adaptive_rep_on=0), dict(ST, adaptive_rep_on=0, pose_rep_cb0_speaking=2.0, pose_rep_light_speaking=2.0)),
         ("adapt_on_equal_static", dict(ST, adaptive_rep_on=1), dict(ST, adaptive_rep_on=1, **NEWEQ))]
ACTIVE = ("adapt_on_speaking_only", dict(adaptive_rep_on=1, pose_rep_cb0_speaking=1.6, pose_rep_light_speaking=1.4))


def main():
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    z = np.load(CONV)
    pre = {"P": int(z["pre_ids_0"].shape[1]), "ids": z["pre_ids_0"], "depth": z["pre_depth_0"], "present": z["pre_present_0"]}
    A = z["stem_1"].astype(np.float32)
    P = pre["P"]
    with torch.no_grad():
        m, ldev, cfgs, info = LN.build_model(CK, devs)

        def run(e, params, seed=777):
            e.start(pre, seed, params)
            e.set_params(params)
            wrote, toks, codes, pose = 0, [], [], []
            for f in range(NF):
                T = P + f
                need = (f + 1) * SPF + LOOK
                e.write_a(wrote, A[wrote:need]); wrote = need
                dec = e.step(T)
                tokB, _, _, _ = e.result(T, dec)
                toks.append(int(tokB))
                if f >= CD:
                    codes.append(list(e.last_codes)); pose.append(list(e.last_pose))
            return {"tok": toks, "codes": codes, "pose": pose}
        outs = {}
        eo = LO.LivePPM(m, ldev, cfgs, devs, max_s=60.0)
        eo.capture()
        for name, po, _ in CASES:
            outs[name] = run(eo, po)
        del eo
        torch.cuda.empty_cache()
        en = LN.LivePPM(m, ldev, cfgs, devs, max_s=60.0)
        en.capture()
        res = {"ck": CK, "frames": NF, "cases": {}}
        for name, po, pn in CASES:
            r = run(en, pn)
            ok = r == outs[name]
            res["cases"][name] = {"old": po, "new": pn, "byte_identical": ok}
            print(f"[adapt] {name}: byte-identical {ok}", flush=True)
        name, pn = ACTIVE
        r = run(en, pn)
        d = float(np.mean(np.asarray(r["pose"]) != np.asarray(outs["defaults"]["pose"])))
        res["cases"][name] = {"new": pn, "pose_frac_differs_from_defaults": d}
        print(f"[adapt] {name}: pose differs from defaults {d:.3f}", flush=True)
        res["all_identical"] = all(v.get("byte_identical", True) for v in res["cases"].values())
        json.dump(res, open(OUT, "w"), indent=1)
        print("ALL IDENTICAL" if res["all_identical"] else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
