#!/usr/bin/env python3
"""posefix 10-26: POSE CFG ONLY WHILE SPEAKING (user: "dynamic speech cfg setting where it's only [when] it speaks") — fixed-seed proof for
BOTH engines. live: deployed aloe-posedup-live/live_ppm_pose.py vs stage-pcfgspk (conv c1 prompt + partner audio); mono: deployed
aloe-pose-monoplex-infer fast_ppm vs stage-pcfgspk/mono (Person 01 clip, channel A empty). Graphs, free generation, same seed:
  knob off (defaults; pose CFG on, speaking_only 0) -> lane / Mimi / pose byte-equal to the old engine;
  knob on -> guided frames == SPEAKING frames recomputed from B's lane (word piece in the last speak_window_frames conversation frames).
usage: python test_pcfgspk.py live|mono <ck> <out json> [frames]"""
import importlib.util
import json
import os
import sys

KIND, CK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NF = int(sys.argv[4]) if len(sys.argv) > 4 else 150
CFG = {"pose_cfg_on": 1, "pose_cfg_s": 3.0, "pose_cfg_cb0": 1, "pose_cfg_light": 1}
CASES = [("defaults", {}, {}), ("cfg_on", dict(CFG), dict(CFG, pose_cfg_speaking_only=0))]
ON = ("cfg_on_speaking_only", dict(CFG, pose_cfg_speaking_only=1))
if KIND == "live":
    E = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/engpcfg"
    sys.path.insert(0, E)
    import live_ppm_pose as NEW  # noqa: E402
    assert os.path.dirname(os.path.realpath(NEW.__file__)).endswith("stage-pcfgspk"), NEW.__file__
    _sp = importlib.util.spec_from_file_location("eng_old", "/data/home/amartyavarma/ppm/aloe-posedup-live/live_ppm_pose.py")
else:
    MI = "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer"
    NEWDIR = "/data/home/amartyavarma/ppm/aloe-posedup-live/stage-pcfgspk/mono"
    sys.path.insert(0, MI); sys.path.insert(0, f"{MI}/ppm"); sys.path.insert(0, NEWDIR)
    os.environ.setdefault("PPM_CD", "2")
    import fast_ppm as NEW  # noqa: E402
    assert os.path.dirname(os.path.realpath(NEW.__file__)) == NEWDIR, NEW.__file__
    import infer as MIF  # noqa: E402
    import ppm_io as IO  # noqa: E402
    _sp = importlib.util.spec_from_file_location("eng_old", f"{MI}/ppm/fast_ppm.py")
OLD = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(OLD)
import numpy as np  # noqa: E402
import torch  # noqa: E402
CD = NEW.CD


def speaking_frames(lane_conv, TE, W):
    """lane_conv [n] = B's lane from conversation frame 0 -> # frames f >= CD whose last W conversation frames hold a word piece (id < TE)."""
    lane_conv = np.asarray(lane_conv)
    return int(sum(bool((lane_conv[max(0, f - W):f] < TE).any()) for f in range(CD, len(lane_conv))))


def main():
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    res = {"kind": KIND, "ck": CK, "frames": NF, "cases": {}}
    with torch.no_grad():
        if KIND == "live":
            z = np.load("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c1.npz")
            pre = {"P": int(z["pre_ids_0"].shape[1]), "ids": z["pre_ids_0"], "depth": z["pre_depth_0"], "present": z["pre_present_0"]}
            A = z["stem_1"].astype(np.float32)
            P = pre["P"]
            SPF, LOOK = NEW.SPF, NEW.VS.LOOK24
            m, ldev, cfgs, info = NEW.build_model(CK, devs)
            mk = lambda M: M.LivePPM(m, ldev, cfgs, devs, max_s=60.0)  # noqa: E731

            def run(e, params, seed=777):
                e.start(pre, seed, params); e.set_params(params)
                wrote, toks, codes, pose = 0, [], [], []
                for f in range(NF):
                    T = P + f
                    need = (f + 1) * SPF + LOOK
                    e.write_a(wrote, A[wrote:need]); wrote = need
                    dec = e.step(T)
                    toks.append(int(e.result(T, dec)[0]))
                    if f >= CD:
                        codes.append(list(e.last_codes)); pose.append(list(e.last_pose))
                g = int(e.pcfg_guided.item()) if getattr(e, "pcfg_guided", None) is not None else None
                return {"tok": toks, "codes": codes, "pose": pose}, g, toks
        else:
            torch.cuda.set_device(devs[-1])
            m, ldev, cfgs, binfo = NEW.build_model(CK, devs)
            mimi = IO.Mimi(devs[-1])
            max_s = (NF + 8) / 12.5 + 2
            nfr = int(round(max_s * 12.5)) + 8
            sil = mimi.encode(np.zeros((2048 + nfr) * 1920, np.float32))[:, :2048 + nfr]
            from safetensors.numpy import load_file
            clip = np.asarray(load_file("/data/home/amartyavarma/video/demo-data/prompts/01_ab63d745/gt/mimi_local.safetensors")["codes"], np.int64)[:, :125]
            x = MIF.assemble(MIF.system_block(clip, None), np.full(NF + CD, NEW.PAD, np.int64), np.zeros((NEW.DEPTH, 0), np.int64),
                             np.zeros((NEW.DEPTH, 0), np.int64), NF, 0)
            P = int(x["P"])
            BASE = {k: v for k, v in dict(MIF.TONIGHT, pose_light_topk=151).items() if k not in ("pose_rep", "pose_rep_speaking", "pose_rep_silent")}
            mk = lambda M: M.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=True, max_s=max_s, mimi_stream=False, silence=sil)  # noqa: E731

            def run(e, params, seed=4242):
                e.start([dict(x, a_mode="empty")], seed, dict(BASE, **params), f0=0)
                for f in range(NF + CD):
                    e.step(f)
                S = e.readback(0, P + NF + CD)
                out = {k: np.asarray(S[k]).tolist() for k in ("lane_b", "dep_b", "pdep")}
                g = int(e.pcfg_guided[0].item()) if getattr(e, "pcfg_guided", None) is not None else None
                return out, g, np.asarray(S["lane_b"])[P:P + NF + CD].tolist()
        eo = mk(OLD); eo.capture()
        outs = {n: run(eo, po)[0] for n, po, _ in CASES}
        del eo
        torch.cuda.empty_cache()
        en = mk(NEW); en.capture()
        for n, po, pn in CASES:
            o, g, _ = run(en, pn)
            res["cases"][n] = {"new": pn, "byte_identical": o == outs[n], "cfg_guided_frames": g}
            print(f"[pcfgspk {KIND}] {n}: byte-identical {o == outs[n]} guided {g}", flush=True)
        n, pn = ON
        o, g, lane = run(en, pn)
        W = int(NEW.DEFAULTS.get("speak_window_frames", 25))
        sp = speaking_frames(lane, NEW.TE, W)
        res["cases"][n] = {"new": pn, "cfg_guided_frames": g, "speaking_frames_from_lane": sp, "frames_total": len(lane) - CD,
                           "pose_differs_from_cfg_on": o != outs["cfg_on"]}
        print(f"[pcfgspk {KIND}] {n}: guided {g} speaking(lane) {sp} of {len(lane) - CD}", flush=True)
        res["all_identical"] = all(v.get("byte_identical", True) for v in res["cases"].values())
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"[pcfgspk {KIND}] " + ("ALL IDENTICAL" if res["all_identical"] else "MISMATCH"), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
