#!/usr/bin/env python3
"""posefix COMPILE check (10-26, user via main: "ultra compile everything"): the STAGED compiled engine (stage-compile via
posefix/sim/engcompile) captured twice in one process, uncompiled (LP.COMPILE False) then compiled, on the same model / GPUs:
(1) teacher-forced, graphed: held-out convs (B's lane + Mimi + pose forced per step, the partner's stem as A): per-frame text logits (zt row
    0), speech cb0 logits (z0 row 0), pose cb0 logits (zp0 row 0), compiled vs uncompiled: max |d|, top-1 agreement, CE vs the real tokens;
(2) free generation, same seed: lane / Mimi / pose equal fraction, compiled vs uncompiled; graphed ms / frame p50 / p95;
(3) light decoders: teacher_logits uncompiled vs torch.compile'd (same mode) on hv / zb recorded by an eager forced pass: max |d|, top-1, CE;
(4) compile wall time (capture() of the compiled engine; cold or warm per the inductor cache state).
usage: python test_compile.py <ck> <out json> <frames per conv> <conv npz> [...]"""
import json
import os
import sys
import time

E = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/engcompile"
sys.path.insert(0, E)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import live_ppm_pose as LP  # noqa: E402

assert os.path.dirname(os.path.realpath(LP.__file__)).endswith("stage-compile"), LP.__file__
CK, OUT, NF, CONVS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
SPF, CD, OFF, CBS = LP.SPF, LP.CD, LP.OFF, LP.CBS
LOOK = LP.VS.LOOK24
NG = 150


def conv_inputs(path, K=0):
    z = np.load(path)
    pre = {"P": int(z[f"pre_ids_{K}"].shape[1]), "ids": z[f"pre_ids_{K}"], "depth": z[f"pre_depth_{K}"], "present": z[f"pre_present_{K}"]}
    return pre, z[f"laneB_{K}"], z[f"depB_{K}"], z[f"pdepB_{K}"], z[f"stem_{1 - K}"].astype(np.float32)


def forced_run(eng, cp, seed):
    """graphed teacher-forced pass -> per-frame logits (CPU fp32) + targets."""
    pre, lane, dep, pdep, A = conv_inputs(cp)
    P = pre["P"]
    eng.start(pre, seed)
    eng.write_a(0, A[:(NF + 2) * SPF + LOOK])
    out = {"zt": [], "yt": [], "z0": [], "y0": [], "zp": [], "yp": []}
    for f in range(NF):
        T = P + f
        frc = (int(lane[f]), dep[f] if dep[f, 0] >= 0 else None, pdep[f] if pdep[f, 0] >= 0 else None)
        dec = eng.step(T, graphs=True, force=frc)
        eng.result(T, dec)
        if f >= 1:
            out["zt"].append(eng.zt[0].float().cpu()); out["yt"].append(int(lane[f]))
        if T >= P + CD and dep[f, 0] >= 0:
            out["z0"].append(eng.z0[0].float().cpu()); out["y0"].append(int(dep[f, 0] - OFF[0]))
        if T >= P + CD and pdep[f, 0] >= 0:
            out["zp"].append(eng.zp0[0].float().cpu()); out["yp"].append(int(pdep[f, 0] - OFF[0]))
    return {k: (torch.stack(v) if k[0] == "z" else torch.tensor(v)) for k, v in out.items()}


def free_run(eng, cp, seed=777):
    pre, lane, dep, pdep, A = conv_inputs(cp)
    P = pre["P"]
    eng.start(pre, seed)
    wrote, ms, toks, codes, pose = 0, [], [], [], []
    for f in range(NG):
        T = P + f
        need = (f + 1) * SPF + LOOK
        eng.write_a(wrote, A[wrote:need]); wrote = need
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dec = eng.step(T, graphs=True)
        tokB, _, _, _ = eng.result(T, dec)
        ms.append(1e3 * (time.perf_counter() - t0))
        toks.append(int(tokB))
        if f >= CD:
            codes.append(list(eng.last_codes)); pose.append(list(eng.last_pose))
    return {"ms": ms, "tok": np.asarray(toks), "codes": np.asarray(codes), "pose": np.asarray(pose)}


def cmp(za, zb, y):
    pa, pb = torch.softmax(za, -1), torch.softmax(zb, -1)
    fin = torch.isfinite(za) & torch.isfinite(zb)
    d = torch.where(fin, (za - zb).abs(), torch.zeros_like(za))
    return {"n": int(len(y)), "maxabs": float(d.max()), "top1_agree": float((za.argmax(-1) == zb.argmax(-1)).float().mean()),
            "tv": float(0.5 * (pa - pb).abs().sum(-1).mean()), "bitwise_equal_frac": float((za == zb).all(-1).float().mean()),
            "ce_uncompiled": float(F.cross_entropy(za, y)), "ce_compiled": float(F.cross_entropy(zb, y))}


@torch.no_grad()
def main():
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    m, ldev, cfgs, info = LP.build_model(CK, devs)
    eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=60.0)
    out = {"ck": CK, "convs": CONVS, "frames_per_conv": NF, "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
           "cache_files_before": sum(len(fs) for _, _, fs in os.walk(os.environ["TORCHINDUCTOR_CACHE_DIR"]))}
    if os.environ.get("SKIP_DEC") != "1":                               # (3) done once: posefix/compile/light_speech_decoder_compile.txt
        # (3) first: eager forced pass to record the decoders' inputs (uncompiled engine, no graphs needed)
        rec = {"speech": [], "pose": []}
        og_s, og_p = eng.light.gen, eng.dec_p.gen

        def rec_s(hv, c0, *a, **k):
            rec["speech"].append([hv.clone(), None]); return og_s(hv, c0, *a, **k)

        def rec_p(hv, c0, *a, **k):
            zb = k.get("zb")
            rec["pose"].append([hv.clone(), None if zb is None else zb.clone()]); return og_p(hv, c0, *a, **k)
        eng.light.gen, eng.dec_p.gen = rec_s, rec_p
        pre, lane, dep, pdep, A = conv_inputs(CONVS[0])
        P = pre["P"]
        eng.start(pre, 99)
        eng.write_a(0, A[:(122) * SPF + LOOK])
        codes_s, codes_p = [], []
        for f in range(120):
            T = P + f
            frc = (int(lane[f]), dep[f] if dep[f, 0] >= 0 else None, pdep[f] if pdep[f, 0] >= 0 else None)
            eng.step(T, graphs=False, force=frc)
            eng.result(T, T >= P + CD)
            if T >= P + CD:
                codes_s.append(dep[f]); codes_p.append(pdep[f])
        eng.light.gen, eng.dec_p.gen = og_s, og_p
        # (3) the speech decoder alone (the engine keeps decoder loops uncompiled by default, COMPILE_DEC=0): teacher_logits uncompiled vs
        # torch.compile'd on the recorded frames (numerics), and one frame CUDA-graphed both ways (ms / frame of the decoder loop)
        nd, dev = eng.light, eng.dl
        t0 = time.time()
        ctl = LP.compiled(nd.teacher_logits)
        za, zb_, ys = [], [], []
        with torch.cuda.device(dev):
            for (hv, zb), c in zip(rec["speech"], codes_s):
                if c[0] < 0:
                    continue
                ct = torch.as_tensor((c - OFF)[None], device=dev)
                za.append(nd.teacher_logits(hv, ct)[0].cpu()); zb_.append(ctl(hv, ct)[0].cpu()); ys.append(ct[0, 1:].cpu())
        za, zb_, y = torch.stack(za), torch.stack(zb_), torch.stack(ys)
        out["light_speech"] = dict(cmp(za.reshape(-1, CBS), zb_.reshape(-1, CBS), y.reshape(-1)), compile_plus_run_s=round(time.time() - t0, 1))
        hv0, ct0 = rec["speech"][5][0].clone(), torch.as_tensor((codes_s[5] - OFF)[None], device=dev)
        for tag, fn in (("uncompiled", nd.teacher_logits), ("compiled", ctl)):
            with torch.cuda.device(dev):
                st = torch.cuda.Stream(dev)
                st.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(st):
                    for _ in range(2):
                        fn(hv0, ct0)
                torch.cuda.current_stream(dev).wait_stream(st)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=st):
                    fn(hv0, ct0)
                torch.cuda.synchronize(dev)
                ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ev0.record()
                for _ in range(50):
                    g.replay()
                ev1.record()
                torch.cuda.synchronize(dev)
                out["light_speech"][f"graphed_ms_{tag}"] = round(ev0.elapsed_time(ev1) / 50, 3)
        LP.log(f"light speech: {json.dumps(out['light_speech'])}")
    json.dump(out, open(OUT, "w"), indent=1)
    # (1) + (2): uncompiled graphs, then compiled graphs
    res = {}
    for tag, on in (("uncompiled", False), ("compiled", True)):
        LP.COMPILE = on
        t0 = time.time()
        eng.capture()
        out[f"capture_s_{tag}"] = round(time.time() - t0, 1)
        LP.log(f"{tag}: capture {out[f'capture_s_{tag}']} s")
        out[f"compile_status_{tag}"] = dict(LP.COMPILE_STATUS)
        LP.log(f"{tag}: status {out[f'compile_status_{tag}']}")
        res[tag] = {"forced": [forced_run(eng, cp, 1234 + i) for i, cp in enumerate(CONVS)], "free": free_run(eng, CONVS[0]),
                    "free2": free_run(eng, CONVS[0])}
        fr = res[tag]["free2"]["ms"][10:]
        out[f"ms_{tag}"] = {"p50": float(np.percentile(fr, 50)), "p95": float(np.percentile(fr, 95)), "mean": float(np.mean(fr))}
        LP.log(f"{tag}: ms {out[f'ms_{tag}']}")
        json.dump(out, open(OUT, "w"), indent=1)
    out["cache_files_after"] = sum(len(fs) for _, _, fs in os.walk(os.environ["TORCHINDUCTOR_CACHE_DIR"]))
    for key, zk, yk in (("text", "zt", "yt"), ("speech_cb0", "z0", "y0"), ("pose_cb0", "zp", "yp")):
        za = torch.cat([r[zk] for r in res["uncompiled"]["forced"]]); zb = torch.cat([r[zk] for r in res["compiled"]["forced"]])
        y = torch.cat([r[yk] for r in res["uncompiled"]["forced"]])
        out[f"forced_{key}"] = cmp(za, zb, y)
        LP.log(f"forced {key}: {json.dumps(out[f'forced_{key}'])}")
    fa, fb = res["uncompiled"]["free"], res["compiled"]["free"]
    out["free_gen_equal"] = {"lane": float((fa["tok"] == fb["tok"]).mean()), "codes": float((fa["codes"] == fb["codes"]).mean()),
                             "pose": float((fa["pose"] == fb["pose"]).mean()),
                             "first_lane_diff_frame": int(np.argmax(fa["tok"] != fb["tok"])) if (fa["tok"] != fb["tok"]).any() else None,
                             "compiled_repeat_equal": bool((res["compiled"]["free"]["codes"] == res["compiled"]["free2"]["codes"]).all())}
    LP.log(f"free gen: {json.dumps(out['free_gen_equal'])}")
    json.dump(out, open(OUT, "w"), indent=1)
    # the op wrappers alone (multinomial1 / fa3_kvcache, COMPILE off) vs the stage-exact engine: must be bit-identical
    import importlib.util
    sp = importlib.util.spec_from_file_location("lp_exact", "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/engexact/live_ppm_pose.py")
    LXE = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(LXE)
    del eng
    torch.cuda.empty_cache()
    ex = LXE.LivePPM(m, ldev, cfgs, devs, max_s=60.0)
    ex.capture()
    fx = free_run(ex, CONVS[0])
    out["uncompiled_vs_stage_exact_equal"] = {"lane": bool((fx["tok"] == fa["tok"]).all()), "codes": bool((fx["codes"] == fa["codes"]).all()),
                                              "pose": bool((fx["pose"] == fa["pose"]).all())}
    out["ms_stage_exact"] = {"p50": float(np.percentile(fx["ms"][10:], 50)), "p95": float(np.percentile(fx["ms"][10:], 95))}
    LP.log(f"uncompiled vs stage-exact: {json.dumps(out['uncompiled_vs_stage_exact_equal'])} {out['ms_stage_exact']}")
    json.dump(out, open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
