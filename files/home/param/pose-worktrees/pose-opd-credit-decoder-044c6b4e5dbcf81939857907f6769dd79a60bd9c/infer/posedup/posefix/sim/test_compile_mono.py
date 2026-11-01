#!/usr/bin/env python3
"""posefix COMPILE check for the MONOPLEX playground engine (stage-compile/mono/fast_ppm.py) on export-357 = test_compile.py for FastPPM:
uncompiled vs compiled graphs in one process (teacher-forced per-frame text / speech cb0 / pose cb0 logits, free-generation equality,
graphed ms / frame), light decoders' teacher_logits uncompiled vs compiled on recorded inputs, and the uncompiled engine vs stage-exact's.
usage: python test_compile_mono.py <ck> <out json> <frames per conv> <conv npz> [...]"""
import importlib.util
import json
import os
import sys
import time

MI = "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer"
NEWDIR = "/data/home/amartyavarma/ppm/aloe-posedup-live/stage-compile/mono"
sys.path.insert(0, MI)
sys.path.insert(0, f"{MI}/ppm")
sys.path.insert(0, NEWDIR)
os.environ.setdefault("PPM_CD", "2")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import fast_ppm as FP  # noqa: E402
assert os.path.dirname(os.path.realpath(FP.__file__)) == NEWDIR, FP.__file__
import infer as MIF  # noqa: E402
import ppm_io as IO  # noqa: E402
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim")
from mono import MONO_PARAMS  # noqa: E402

CK, OUT, NF, CONVS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
CD, OFF, PAD, DEPTH, CBS, SPF = FP.CD, FP.OFF, FP.PAD, FP.DEPTH, FP.CBS, 1920
NG = 150


def conv_x(path, n, K=0):
    zc = np.load(path)
    d1 = zc[f"pre_depth_{K}"][1]
    clip = (d1[d1[:, 0] >= 0] - OFF).T.astype(np.int64)
    sp = MIF.system_block(clip, None)
    lane = np.full(n + CD, PAD, np.int64)
    x = MIF.assemble(sp, lane, np.zeros((DEPTH, 0), np.int64), np.zeros((DEPTH, 0), np.int64), n, 0)
    return x, zc[f"laneB_{K}"], zc[f"depB_{K}"], zc[f"pdepB_{K}"]


def frc_of(lane, dep, pdep, f):
    frc = {"lane": [int(lane[f])]}
    if dep[f, 0] >= 0:
        frc["codes"] = dep[f][None]
    if pdep[f, 0] >= 0:
        frc["pose"] = pdep[f][None]
    return frc


def forced_run(eng, cp, seed):
    x, lane, dep, pdep = conv_x(cp, NF)
    eng.start([dict(x, a_mode="empty")], seed, dict(MONO_PARAMS), f0=0)
    out = {"zt": [], "yt": [], "z0": [], "y0": [], "zp": [], "yp": []}
    for f in range(NF):
        eng.step(f, graphs=True, force=frc_of(lane, dep, pdep, f), cold_pad=False)
        torch.cuda.synchronize()
        if f >= 1:
            out["zt"].append(eng.zt[0].float().cpu()); out["yt"].append(int(lane[f]))
        if f >= CD and dep[f, 0] >= 0:
            out["z0"].append(eng.z0[0].float().cpu()); out["y0"].append(int(dep[f, 0] - OFF[0]))
        if f >= CD and pdep[f, 0] >= 0:
            out["zp"].append(eng.zp0[0].float().cpu()); out["yp"].append(int(pdep[f, 0] - OFF[0]))
    return {k: (torch.stack(v) if k[0] == "z" else torch.tensor(v)) for k, v in out.items()}


def free_run(eng, cp):
    x, lane, dep, pdep = conv_x(cp, NG)
    eng.start([dict(x, a_mode="empty")], 777, dict(MONO_PARAMS), f0=0)
    ms = []
    for f in range(NG + CD):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.step(f, graphs=True)
        torch.cuda.synchronize()
        ms.append(1e3 * (time.perf_counter() - t0))
    S = eng.readback(0, int(x["P"]) + NG + CD)
    return {"ms": ms, "lane": S["lane_b"], "dep": S["dep_b"], "pdep": S["pdep"]}


def cmp(za, zb, y):
    pa, pb = torch.softmax(za, -1), torch.softmax(zb, -1)
    fin = torch.isfinite(za) & torch.isfinite(zb)
    d = torch.where(fin, (za - zb).abs(), torch.zeros_like(za))
    return {"n": int(len(y)), "maxabs": float(d.max()), "top1_agree": float((za.argmax(-1) == zb.argmax(-1)).float().mean()),
            "tv": float(0.5 * (pa - pb).abs().sum(-1).mean()), "bitwise_equal_frac": float((za == zb).all(-1).float().mean()),
            "ce_uncompiled": float(F.cross_entropy(za, y)), "ce_compiled": float(F.cross_entropy(zb, y))}


def main():
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    torch.cuda.set_device(devs[-1])
    out = {"ck": CK, "convs": CONVS, "frames_per_conv": NF}
    with torch.inference_mode():
        m, ldev, cfgs, binfo = FP.build_model(CK, devs)
        mimi = IO.Mimi(devs[-1])
        max_s = (NF + 8) / 12.5 + 2
        p_max = 2048
        nfr = int(round(max_s * 12.5)) + 8
        sil = mimi.encode(np.zeros((p_max + nfr) * SPF, np.float32))[:, :p_max + nfr]
        eng = FP.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=True, max_s=max_s, mimi_stream=False, silence=sil)
        if os.environ.get("MONO_DEC_TEST") == "1":                     # decoder loops are uncompiled by default (FASTPPM_COMPILE_DEC=0)
            rec = {"speech": [], "pose": []}
            og_s, og_p = eng.dec_s.gen, eng.dec_p.gen

            def rec_s(hv, c0, *a, **k):
                rec["speech"].append([hv.clone(), None]); return og_s(hv, c0, *a, **k)

            def rec_p(hv, c0, *a, **k):
                zb = k.get("zb")
                rec["pose"].append([hv.clone(), None if zb is None else zb.clone()]); return og_p(hv, c0, *a, **k)
            eng.dec_s.gen, eng.dec_p.gen = rec_s, rec_p
            x, lane, dep, pdep = conv_x(CONVS[0], 120)
            eng.start([dict(x, a_mode="empty")], 99, dict(MONO_PARAMS), f0=0)
            cs, cp_ = [], []
            for f in range(120):
                eng.step(f, graphs=False, force=frc_of(lane, dep, pdep, f), cold_pad=False)
                if f >= CD:
                    cs.append(dep[f]); cp_.append(pdep[f])
            torch.cuda.synchronize()
            eng.dec_s.gen, eng.dec_p.gen = og_s, og_p
            for key, nd, codes in (("speech", eng.dec_s, cs), ("pose", eng.dec_p, cp_)):
                ctl = FP.compiled(nd.teacher_logits)
                za, zb_, ys = [], [], []
                t0 = time.time()
                for (hv, zb), c in zip(rec[key], codes):
                    if c[0] < 0:
                        continue
                    ct = torch.as_tensor((c - OFF)[None], device=hv.device)
                    za.append(nd.teacher_logits(hv, ct, zb)[0].cpu()); zb_.append(ctl(hv, ct, zb)[0].cpu()); ys.append(ct[0, 1:].cpu())
                za, zb_, y = torch.stack(za), torch.stack(zb_), torch.stack(ys)
                out[f"light_{key}"] = dict(cmp(za.reshape(-1, CBS), zb_.reshape(-1, CBS), y.reshape(-1)), compile_plus_run_s=round(time.time() - t0, 1))
                print(f"[mono-compile] light {key}: {json.dumps(out[f'light_{key}'])}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
        res = {}
        for tag, on in (("uncompiled", False), ("compiled", True)):
            FP.COMPILE = on
            t0 = time.time()
            eng.capture()
            out[f"capture_s_{tag}"] = round(time.time() - t0, 1)
            out[f"compile_status_{tag}"] = dict(FP.COMPILE_STATUS)
            print(f"[mono-compile] {tag}: status {out[f'compile_status_{tag}']}", flush=True)
            res[tag] = {"forced": [forced_run(eng, c, 1234 + i) for i, c in enumerate(CONVS)], "free": free_run(eng, CONVS[0]),
                        "free2": free_run(eng, CONVS[0])}
            fr = res[tag]["free2"]["ms"][10:]
            out[f"ms_{tag}"] = {"p50": float(np.percentile(fr, 50)), "p95": float(np.percentile(fr, 95)), "mean": float(np.mean(fr))}
            print(f"[mono-compile] {tag}: capture {out[f'capture_s_{tag}']} s ms {out[f'ms_{tag}']}", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
        for key, zk, yk in (("text", "zt", "yt"), ("speech_cb0", "z0", "y0"), ("pose_cb0", "zp", "yp")):
            za = torch.cat([r[zk] for r in res["uncompiled"]["forced"]]); zb = torch.cat([r[zk] for r in res["compiled"]["forced"]])
            y = torch.cat([r[yk] for r in res["uncompiled"]["forced"]])
            out[f"forced_{key}"] = cmp(za, zb, y)
            print(f"[mono-compile] forced {key}: {json.dumps(out[f'forced_{key}'])}", flush=True)
        fa, fb = res["uncompiled"]["free"], res["compiled"]["free"]
        out["free_gen_equal"] = {"lane": float((fa["lane"] == fb["lane"]).mean()), "codes": float((fa["dep"] == fb["dep"]).mean()),
                                 "pose": float((fa["pdep"] == fb["pdep"]).mean()),
                                 "compiled_repeat_equal": bool((res["compiled"]["free"]["dep"] == res["compiled"]["free2"]["dep"]).all())}
        print(f"[mono-compile] free gen: {json.dumps(out['free_gen_equal'])}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
        sp = importlib.util.spec_from_file_location("fp_exact", "/data/home/amartyavarma/ppm/aloe-posedup-live/stage-exact/mono/fast_ppm.py")
        FE = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(FE)
        del eng
        torch.cuda.empty_cache()
        ex = FE.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=True, max_s=max_s, mimi_stream=False, silence=sil)
        ex.capture()
        fx = free_run(ex, CONVS[0])
        out["uncompiled_vs_stage_exact_equal"] = {"lane": bool((fx["lane"] == fa["lane"]).all()), "codes": bool((fx["dep"] == fa["dep"]).all()),
                                                  "pose": bool((fx["pdep"] == fa["pdep"]).all())}
        out["ms_stage_exact"] = {"p50": float(np.percentile(fx["ms"][10:], 50)), "p95": float(np.percentile(fx["ms"][10:], 95))}
        print(f"[mono-compile] uncompiled vs stage-exact: {json.dumps(out['uncompiled_vs_stage_exact_equal'])} {out['ms_stage_exact']}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
