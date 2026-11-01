#!/usr/bin/env python3
"""Is the live engine's fixed-seed output reproducible ACROSS PROCESSES (= backend restarts)? light_kernels.gemv (the fused 0.6B speech and pose
decoders) is @triton.autotune'd: each process times its configs (different BK = different fp32 reduction order) and keeps the fastest.
One process: random weights (export-498 architecture), engine = OLD (the pre-posecfg backup) or NEW (staged, pose CFG off), one 240-frame
session (seed 4242, defaults); saves the outputs + the autotuner's chosen configs; then (sensitivity) the same session EAGER with the chosen
configs and EAGER with every gemv key forced to another config.  usage: autotune_test.py <old|new> <ci.pkl> <out.npz>"""
import json, pickle, sys
STAGE = "/data/home/amartyavarma/ppm/aloe-posedup-live/stage-posecfg"
OLD = "/data/home/amartyavarma/ppm/aloe-posedup-live/live_ppm_pose.py.bak-pre-posecfg-1025"
sys.path.insert(0, STAGE)
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/verify/posecfg-1025/dev")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import live_ppm_pose as LPn  # noqa: E402
WHICH, CI, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
_argv, sys.argv = sys.argv, [sys.argv[0], "-", "-", "240"]
import engine_test as ET  # noqa: E402   (rand_fill, run, cmp; loads the OLD module as ET.LPo)
sys.argv = _argv
LP = ET.LPo if WHICH == "old" else LPn
assert LP.__file__ == (OLD if WHICH == "old" else STAGE + "/live_ppm_pose.py"), LP.__file__
LK = LP.LK


def cfgs():
    return {f"{n}:{k}": str(c) for n, f in (("gemv", LK._gemv_k), ("gemv_small", LK._gemv_small_k)) for k, c in f.cache.items()}


with torch.no_grad():
    ci = pickle.load(open(CI, "rb"))
    LP.ckpt_info = lambda p: dict(ci)
    LP.load_weights = ET.rand_fill
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    m, ldev, c3, info = LP.build_model(ci["path"], devs)
    z = np.load(ET.CONV)
    pre = {"P": int(z["pre_ids_0"].shape[1]), "ids": z["pre_ids_0"], "depth": z["pre_depth_0"], "present": z["pre_present_0"]}
    A = z["stem_1"].astype(np.float32)
    eng = LP.LivePPM(m, ldev, c3, devs, max_s=600.0)
    eng.capture()
    o = ET.run(eng, pre, A, {}, 240)
    chosen = cfgs()
    oe = ET.run(eng, pre, A, {}, 60, graphs=False)
    for f in (LK._gemv_k, LK._gemv_small_k):                     # force another config for every key
        for k, c in list(f.cache.items()):
            f.cache[k] = next(x for x in f.configs if str(x) != str(c))
    of = ET.run(eng, pre, A, {}, 60, graphs=False)
    r = {"which": WHICH, "configs": chosen, "eager_vs_graph_60": ET.cmp(ET.trunc(o, 60), oe), "forced_vs_chosen_eager_60": ET.cmp(oe, of)}
    np.savez(OUT, tok=np.asarray(o["tok"]), mimi=np.asarray(o["mimi"]), pose=np.asarray(o["pose"]), yb=np.frombuffer(o["yb"], np.uint8),
             meta=json.dumps(r, default=str))
    print("AUTOTUNE", json.dumps(r, default=str))
