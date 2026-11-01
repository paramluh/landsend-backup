#!/usr/bin/env python3
"""posefix 10-26: pose rep-pen split (user: separate cb0 and acoustic) — fixed-seed proof for the monoplex engine. Old = the deployed
aloe-pose-monoplex-infer/ppm/fast_ppm.py (combined pose_rep), new = stage-reppen/mono/fast_ppm.py (pose_rep_cb0 / pose_rep_light + adaptive
variants, old names = alias to both). Same model, same seed, free generation (Person 01 voice clip, no prefill), CUDA graphs:
  identical cases: old {X} vs new {X} (alias) vs new {X written with both new keys} -> lane / Mimi / pose byte-equal
  split case: new cb0-only / light-only penalty -> differs from both (the knobs act separately).
usage: python test_reppen_mono.py <ck> <out json> [frames]"""
import importlib.util
import json
import os
import sys

MI = "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer"
NEWDIR = "/data/home/amartyavarma/ppm/aloe-posedup-live/stage-reppen/mono"
sys.path.insert(0, MI)
sys.path.insert(0, f"{MI}/ppm")
sys.path.insert(0, NEWDIR)
os.environ.setdefault("PPM_CD", "2")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import fast_ppm as FN  # noqa: E402
assert os.path.dirname(os.path.realpath(FN.__file__)) == NEWDIR, FN.__file__
import infer as MIF  # noqa: E402
import ppm_io as IO  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402
_sp = importlib.util.spec_from_file_location("fp_old", f"{MI}/ppm/fast_ppm.py")
FO_ = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(FO_)

CK, OUT = sys.argv[1], sys.argv[2]
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 120
CD, PAD, DEPTH, SPF = FN.CD, FN.PAD, FN.DEPTH, 1920
BASE = {k: v for k, v in dict(MIF.TONIGHT, pose_light_topk=151, pose_cfg_on=0).items()
        if k not in ("pose_rep", "pose_rep_speaking", "pose_rep_silent")}     # = 1.0 defaults; a raw old key would override the new ones
CASES = [  # (name, old params, [new params variants that must equal old])
    ("defaults", {}, [{}]),
    ("static_1.3", {"pose_rep": 1.3, "pose_rep_win": 0}, [{"pose_rep": 1.3, "pose_rep_win": 0},
                                                         {"pose_rep_cb0": 1.3, "pose_rep_light": 1.3, "pose_rep_win": 0}]),
    ("static_1.5_win64", {"pose_rep": 1.5, "pose_rep_win": 64}, [{"pose_rep": 1.5, "pose_rep_win": 64},
                                                                {"pose_rep_cb0": 1.5, "pose_rep_light": 1.5, "pose_rep_win": 64}]),
    ("adaptive", {"adaptive_rep_on": 1, "pose_rep_speaking": 1.4, "pose_rep_silent": 1.2},
     [{"adaptive_rep_on": 1, "pose_rep_speaking": 1.4, "pose_rep_silent": 1.2},
      {"adaptive_rep_on": 1, "pose_rep_cb0_speaking": 1.4, "pose_rep_light_speaking": 1.4, "pose_rep_cb0_silent": 1.2,
       "pose_rep_light_silent": 1.2}]),
]
SPLIT = [("cb0_only_1.5", {"pose_rep_cb0": 1.5, "pose_rep_win": 0}), ("light_only_1.5", {"pose_rep_light": 1.5, "pose_rep_win": 0})]


def main():
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    torch.cuda.set_device(devs[-1])
    with torch.inference_mode():
        m, ldev, cfgs, binfo = FN.build_model(CK, devs)
        mimi = IO.Mimi(devs[-1])
        max_s = (NF + 8) / 12.5 + 2
        p_max = 2048
        nfr = int(round(max_s * 12.5)) + 8
        sil = mimi.encode(np.zeros((p_max + nfr) * SPF, np.float32))[:, :p_max + nfr]
        clip = np.asarray(load_file("/data/home/amartyavarma/video/demo-data/prompts/01_ab63d745/gt/mimi_local.safetensors")["codes"], np.int64)[:, :125]
        sp = MIF.system_block(clip, None)
        lane = np.full(NF + CD, PAD, np.int64)
        x = MIF.assemble(sp, lane, np.zeros((DEPTH, 0), np.int64), np.zeros((DEPTH, 0), np.int64), NF, 0)
        P = int(x["P"])

        def run(e, params, seed=4242):
            e.start([dict(x, a_mode="empty")], seed, dict(BASE, **params), f0=0)
            for f in range(NF + CD):
                e.step(f)
            S = e.readback(0, P + NF + CD)
            return {k: np.asarray(S[k]) for k in ("lane_b", "dep_b", "pdep")}

        def eq(a, b):
            return all(np.array_equal(a[k], b[k]) for k in a)
        res = {"ck": CK, "frames": NF, "cases": {}}
        outs = {}
        eo = FO_.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=True, max_s=max_s, mimi_stream=False, silence=sil)
        eo.capture()
        for name, po, _ in CASES:
            outs[name] = run(eo, po)
        del eo
        torch.cuda.empty_cache()
        en = FN.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=True, max_s=max_s, mimi_stream=False, silence=sil)
        en.capture()
        for name, po, news in CASES:
            r = [eq(outs[name], run(en, pn)) for pn in news]
            res["cases"][name] = {"old": po, "new_variants": news, "byte_identical": r}
            print(f"[reppen] {name}: byte-identical {r}", flush=True)
        both = outs["static_1.3"]
        for name, pn in SPLIT:
            o = run(en, pn)
            d = {k: float(np.mean(o[k][P:] != outs["defaults"][k][P:])) for k in o}
            res["cases"][name] = {"new": pn, "frac_differs_from_defaults": d, "equals_both": eq(o, both)}
            print(f"[reppen] {name}: differs from defaults {d}", flush=True)
        res["all_identical"] = all(all(v["byte_identical"]) for v in res["cases"].values() if "byte_identical" in v)
        json.dump(res, open(OUT, "w"), indent=1)
        print("ALL IDENTICAL" if res["all_identical"] else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
