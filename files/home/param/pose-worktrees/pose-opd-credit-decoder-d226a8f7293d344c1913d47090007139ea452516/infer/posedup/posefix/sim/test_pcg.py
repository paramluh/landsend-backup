#!/usr/bin/env python3
"""posefix 10-26: per-frame 'pcg' reporting (pose CFG actually applied, after the speaking gate) on the live engine — proof that it is
reporting only: deployed live_ppm_pose.py vs stage-pcg, same seed / prompt / A audio, graphs; lane / Mimi / pose byte-equal, and the
per-frame last_pcg sums to the engine counter and equals the SPEAKING-gated frames. usage: python test_pcg.py <ck> <out json> [frames]"""
import importlib.util
import json
import os
import sys

E = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/engpcg"
sys.path.insert(0, E)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import live_ppm_pose as NEW  # noqa: E402
assert os.path.dirname(os.path.realpath(NEW.__file__)).endswith("stage-pcg"), NEW.__file__
_sp = importlib.util.spec_from_file_location("eng_old", "/data/home/amartyavarma/ppm/aloe-posedup-live/live_ppm_pose.py")
OLD = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(OLD)
CK, OUT = sys.argv[1], sys.argv[2]
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 150
CFG = {"pose_cfg_on": 1, "pose_cfg_s": 3.0, "pose_cfg_cb0": 1, "pose_cfg_light": 1}
CASES = [("defaults", {}), ("cfg_on", dict(CFG)), ("cfg_on_speaking_only", dict(CFG, pose_cfg_speaking_only=1))]


def main():
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    z = np.load("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c1.npz")
    pre = {"P": int(z["pre_ids_0"].shape[1]), "ids": z["pre_ids_0"], "depth": z["pre_depth_0"], "present": z["pre_present_0"]}
    A = z["stem_1"].astype(np.float32)
    P, SPF, LOOK, CD = pre["P"], NEW.SPF, NEW.VS.LOOK24, NEW.CD
    with torch.no_grad():
        m, ldev, cfgs, info = NEW.build_model(CK, devs)

        def run(e, params, seed=777):
            e.start(pre, seed, params); e.set_params(params)
            wrote, toks, codes, pose, pcg = 0, [], [], [], []
            for f in range(NF):
                T = P + f
                need = (f + 1) * SPF + LOOK
                e.write_a(wrote, A[wrote:need]); wrote = need
                dec = e.step(T)
                toks.append(int(e.result(T, dec)[0]))
                pcg.append(int(getattr(e, "last_pcg", -1)))
                if f >= CD:
                    codes.append(list(e.last_codes)); pose.append(list(e.last_pose))
            return {"tok": toks, "codes": codes, "pose": pose}, pcg, int(e.pcfg_guided.item())
        eo = OLD.LivePPM(m, ldev, cfgs, devs, max_s=60.0); eo.capture()
        outs = {n: run(eo, p) for n, p in CASES}
        del eo
        torch.cuda.empty_cache()
        en = NEW.LivePPM(m, ldev, cfgs, devs, max_s=60.0); en.capture()
        res = {"cases": {}}
        for n, p in CASES:
            o, pcg, g = run(en, p)
            r = {"byte_identical": o == outs[n][0], "pcg_sum": int(sum(pcg)), "counter": g, "counter_old": outs[n][2], "pcg_values": sorted(set(pcg))}
            res["cases"][n] = r
            print(f"[pcg] {n}: {json.dumps(r)}", flush=True)
        res["ok"] = all(v["byte_identical"] and v["pcg_sum"] == v["counter"] == v["counter_old"] for v in res["cases"].values())
        json.dump(res, open(OUT, "w"), indent=1)
        print("[pcg] " + ("ALL OK" if res["ok"] else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
