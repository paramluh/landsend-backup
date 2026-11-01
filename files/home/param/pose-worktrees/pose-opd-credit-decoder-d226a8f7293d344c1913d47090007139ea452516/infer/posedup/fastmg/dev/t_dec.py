"""fast decoder vs the live FusedDec (random weights, export-498 architecture): graph latency of gen() (31 codebooks) and teacher-forced
logits agreement. usage: t_dec.py <out json>"""
import json, math, pickle, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/fastmg")
import torch
import live_ppm_pose_mg as MG
import fdec
LP, M = MG.LP, MG.M
BF = torch.bfloat16
d = torch.device("cuda:0")
ci = pickle.load(open("/data/home/amartyavarma/ppm/aloe-posedup-live/verify/posecfg-1025/dev/ci498.pkl", "rb"))
c32, c4, c06 = M.cfg32(ci["layers"]), M.cfg4b(ci["tower_layers"]), M.cfg06(ci["dec_layers"])
torch.set_default_dtype(BF)
with torch.device("meta"):
    m = M.PPM(c32, c4, c06, ci["proj_r"], ci["tower_vocab"], vadapter=LP.VADAPT, nopose=False, decp_bb=True)
torch.set_default_dtype(torch.float32)
dec = m.dec_s
dec.to_empty(device=d)
dec.model.rotary_emb = type(dec.model.rotary_emb)(config=c06, device=d)
g = torch.Generator(device=d).manual_seed(5)
with torch.no_grad():
    for n, p in dec.named_parameters():
        if "norm" in n:
            p.copy_((torch.rand(p.shape, generator=g, device=d) + 0.5).to(p.dtype))
        elif n.endswith(".bias"):
            p.zero_()
        else:
            p.copy_((torch.randn(p.shape, generator=g, device=d) * 0.02).to(p.dtype))
for p in dec.parameters():
    p.requires_grad_(False)
res = {}
Hv = c4.hidden_size
with torch.no_grad(), torch.cuda.device(d):
    for S in (1, 2):
        fd = LP.FusedDec(dec, d) if S == 1 else LP.FusedDecS(dec, d, 2)
        fx = fdec.FastDec(fd, S)
        hv = torch.randn(S, Hv, device=d).to(BF)
        NI = 1
        c0 = torch.randint(0, 2048, (NI,), device=d)
        temp = torch.tensor(1.0, device=d); topk = torch.tensor(30, device=d)
        cw, con = torch.tensor(3.0, device=d), torch.tensor(True, device=d)
        gen = torch.Generator(device=d)
        kw = {} if S == 1 else {"cfg_w": cw, "cfg_on": con}
        out = {}
        for name, obj in (("live", fd), ("fast", fx)):
            fn = lambda: obj.gen(hv, c0, gen, temp, topk, **kw)
            s = torch.cuda.Stream(d); s.wait_stream(torch.cuda.current_stream(d))
            with torch.cuda.stream(s):
                fn(); fn()
            torch.cuda.synchronize(d)
            gr = torch.cuda.CUDAGraph(); gr.register_generator_state(gen)
            with torch.cuda.graph(gr, stream=s):
                o = fn()
            torch.cuda.synchronize(d)
            ts = []
            for _ in range(20):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize(d)
                ts.append(e0.elapsed_time(e1))
            ts.sort()
            out[name] = {"ms_p50": round(ts[10], 3), "ms_min": round(ts[0], 3)}
            # same seed -> codes
            gen.manual_seed(77); gr.replay(); torch.cuda.synchronize(d)
            out[name]["codes"] = o.tolist()
        # teacher-forced logits on the live engine's own codes
        codes = torch.tensor(out["live"]["codes"], device=d)
        if S == 1:
            lt = fd.teacher_logits(hv, codes)                    # [1, 31, CBS]
            x = torch.cat([dec.proj(hv)[:, None].to(BF), dec.cb0_emb(codes[:, 0])[:, None].to(BF)], 1)
            lf = []
            for dd in range(1, LP.DEPTH):
                lf.append(fx.step_head(x, dd - 1, dd).clone())
                if dd < LP.DEPTH - 1:
                    x = dec.model.embed_tokens(codes[:, dd] + (dd - 1) * LP.CBS)[:, None].to(BF)
            lf = torch.stack(lf, 1)
            out["teacher"] = {"max_abs": float((lt - lf).abs().max()), "logit_std": float(lt.std()), "argmax_agree": float((lt.argmax(-1) == lf.argmax(-1)).float().mean()),
                              "bitwise_frac": float((lt == lf).float().mean())}
        out["codes_identical_same_seed"] = out["live"]["codes"] == out["fast"]["codes"]
        out["live"].pop("codes"); out["fast"].pop("codes")
        res[f"S{S}"] = out
        print(S, out, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
