#!/usr/bin/env python3
"""posefix 10-26: light-decoder precision check (main c7: the trainer runs the replicated 0.6B decoders dec_s / dec_p as fp32 masters under
bf16 autocast; the live engines' FusedDec ran pure bf16). Engine = the STAGED trainer-exact engine (stage-exact via posefix/sim/engexact).
(1) TEACHER-FORCED: held-out convs (lane + Mimi + pose of B forced, the partner's stem as A) through the engine (eager steps); the decoders'
    inputs (hv, zb) recorded at every generated frame, then per decoder and frame, cb1..31 logits of
      REF  the trainer's eval light path on an fp32 COPY of the decoder (Decoder.inputs -> HF Qwen3Model (sdpa) -> logits, bf16 autocast;
           = model_ppm.Decoder.forward's eval branch; fp32(bf16 export) = the only weights on disk)
      NEW  the staged engine's FusedDec.teacher_logits (trainer-exact dtype flow)
      OLD  the live engine's FusedDec.teacher_logits (pure bf16; ppm/aloe-posedup-live/live_ppm_pose.py loaded as a second module)
    vs REF: top-1 agreement, TV, max |dlogit|; light CE of each vs the real codes; cb0 CE (tower head, unchanged) for the eval comparison.
    The pose CFG decoder (FusedDecS, 2 rows) is checked row by row against the 1-row NEW logits.
(2) GRAPH == EAGER: free generation from the conv's prompt, same seed, eager steps vs captured graphs: identical codes / lane.
(3) ms / frame: graphed steps, NEW engine vs OLD engine (same model, same GPUs).
usage: python test_light.py <ck> <out json> <frames per conv> <conv npz> [...]"""
import copy
import importlib.util
import json
import os
import sys
import time

E = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/engexact"
sys.path.insert(0, E)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import live_ppm_pose as LP  # noqa: E402

assert os.path.dirname(os.path.realpath(LP.__file__)).endswith("stage-exact"), LP.__file__
_sp = importlib.util.spec_from_file_location("lp_old", "/data/home/amartyavarma/ppm/aloe-posedup-live/live_ppm_pose.py")
LO = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(LO)
assert LO.FusedDec is not LP.FusedDec and not issubclass(LO.FusedDec, LP.FusedDecS)

CK, OUT, NF, CONVS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
SPF, CD, OFF, DEPTH, CBS = LP.SPF, LP.CD, LP.OFF, LP.DEPTH, LP.CBS
LOOK = LP.VS.LOOK24


def ref_logits(dec32, hv, codes_local, zb, dev):
    """trainer eval path on the fp32 copy -> [1, 31, CBS] fp32 (hv [1, Hv] bf16, codes [32] local, zb [1, Hl] bf16 or None)."""
    depth = torch.as_tensor((codes_local + OFF)[None], device=dev)
    flat = torch.zeros(1, dtype=torch.long, device=dev)
    with torch.cuda.device(dev), torch.autocast("cuda", dtype=torch.bfloat16):
        x, _ = dec32.inputs(hv, depth, flat) if zb is None else dec32.inputs(hv, depth, flat, zb)
        hl = dec32.model(inputs_embeds=x, use_cache=False).last_hidden_state
        return dec32.logits(hl).float()


def metr(z, zr, y):
    p, pr = torch.softmax(z, -1), torch.softmax(zr, -1)
    return {"top1_agree": float((z.argmax(-1) == zr.argmax(-1)).float().mean()), "tv": float(0.5 * (p - pr).abs().sum(-1).mean()),
            "maxabs": float((z - zr).abs().max()), "meanabs": float((z - zr).abs().mean()),
            "ce": float(F.cross_entropy(z.reshape(-1, CBS), y.reshape(-1)))}


def conv_inputs(path, K=0):
    z = np.load(path)
    pre = {"P": int(z[f"pre_ids_{K}"].shape[1]), "ids": z[f"pre_ids_{K}"], "depth": z[f"pre_depth_{K}"], "present": z[f"pre_present_{K}"]}
    return pre, z[f"laneB_{K}"], z[f"depB_{K}"], z[f"pdepB_{K}"], z[f"stem_{1 - K}"].astype(np.float32)


@torch.no_grad()
def main():
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    m, ldev, cfgs, info = LP.build_model(CK, devs)
    eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=60.0)
    assert type(eng.light) is LP.FusedDec and type(eng.dec_p) is LP.FusedDec and eng.pcfg and type(eng.dec_pu) is LP.FusedDecS
    old = {"speech": LO.FusedDec(m.dec_s, eng.dl), "pose": LO.FusedDec(m.dec_p, eng.d0)}
    out = {"ck": CK, "convs": CONVS, "frames_per_conv": NF, "eval_498": {"speech_light_ce": None, "pose_light_ce": None}}
    # ---------------------------------------------------------------- (1) teacher-forced
    rec = {"speech": [], "pose": []}
    og_s, og_p = eng.light.gen, eng.dec_p.gen

    def rec_s(hv, c0, *a, **k):
        rec["speech"].append([hv.clone(), None, None]); return og_s(hv, c0, *a, **k)

    def rec_p(hv, c0, *a, **k):
        zb = k.get("zb")
        rec["pose"].append([hv.clone(), None if zb is None else zb.clone(), None]); return og_p(hv, c0, *a, **k)
    eng.light.gen, eng.dec_p.gen = rec_s, rec_p
    for ci, cp in enumerate(CONVS):
        pre, lane, dep, pdep, A = conv_inputs(cp)
        P = pre["P"]
        n0s, n0p = len(rec["speech"]), len(rec["pose"])
        eng.start(pre, 1234 + ci)
        eng.write_a(0, A[:(NF + 2) * SPF + LOOK])
        gen_f = []
        for f in range(NF):
            T = P + f
            frc = (int(lane[f]), dep[f] if dep[f, 0] >= 0 else None, pdep[f] if pdep[f, 0] >= 0 else None)
            eng.step(T, graphs=False, force=frc)
            eng.result(T, T >= P + CD)
            if T >= P + CD:
                gen_f.append(f)
        assert len(rec["speech"]) - n0s == len(gen_f) == len(rec["pose"]) - n0p, (len(rec["speech"]) - n0s, len(rec["pose"]) - n0p, len(gen_f))
        for key, codes_all in (("speech", dep), ("pose", pdep)):
            for r, f in zip(rec[key][-len(gen_f):], gen_f):
                r[2] = codes_all[f] if codes_all[f, 0] >= 0 else None          # frames without real codes: not scored
        LP.log(f"conv {cp}: {len(gen_f)} frames recorded")
    eng.light.gen, eng.dec_p.gen = og_s, og_p
    for key, dec, nd, dev, tw in (("speech", m.dec_s, eng.light, eng.dl, m.tower_s), ("pose", m.dec_p, eng.dec_p, eng.d0, m.tower_p)):
        torch.cuda.set_device(dev)                           # Triton launches on the current device (posefix 10-26)
        with torch.cuda.device(dev):
            dec32 = copy.deepcopy(dec).float()
        Z = {"old": [], "new": [], "ref": [], "y": [], "c0": [], "y0": []}
        for hv, zb, codes in rec[key]:
            if codes is None:
                continue
            cl = (codes - OFF).astype(np.int64)
            ct = torch.as_tensor(cl[None], device=dev)
            Z["ref"].append(ref_logits(dec32, hv, cl, zb, dev)[0])
            Z["new"].append(nd.teacher_logits(hv, ct, zb)[0])
            Z["old"].append(old[key].teacher_logits(hv, ct, zb)[0])
            Z["y"].append(ct[0, 1:])
            Z["c0"].append((hv @ tw.cb0_head.weight.t()).float()[0])
            Z["y0"].append(ct[0, 0])
        zr, y = torch.stack(Z["ref"]), torch.stack(Z["y"])
        res = {"frames": len(Z["y"]), "ref_light_ce": float(F.cross_entropy(zr.reshape(-1, CBS), y.reshape(-1))),
               "cb0_ce": float(F.cross_entropy(torch.stack(Z["c0"]), torch.stack(Z["y0"]))),
               "new_vs_ref": metr(torch.stack(Z["new"]), zr, y), "old_vs_ref": metr(torch.stack(Z["old"]), zr, y)}
        if key == "pose":                                    # pose CFG decoder (2 rows, shares dec_p's weights) row by row vs the 1-row NEW
            n2 = min(64, len(Z["y"]) // 2)
            rows = [(r[0], r[1], r[2]) for r in rec[key] if r[2] is not None][:2 * n2]
            dmax, same = 0.0, 0
            for i in range(n2):
                (h0, z0_, c0_), (h1, z1_, c1_) = rows[2 * i], rows[2 * i + 1]
                ct2 = torch.as_tensor(np.stack([c0_, c1_]) - OFF, device=dev)
                l2 = eng.dec_pu.teacher_logits(torch.cat([h0, h1]), ct2, None if z0_ is None else torch.cat([z0_, z1_]))
                l1 = torch.cat([nd.teacher_logits(h0, ct2[0:1], z0_), nd.teacher_logits(h1, ct2[1:2], z1_)])
                dmax = max(dmax, float((l2 - l1).abs().max()))
                same += int(torch.equal(l2, l1))
            res["cfg2_vs_1row"] = {"pairs": n2, "maxabs": dmax, "bitwise_equal_pairs": same}
        out[key] = res
        LP.log(f"{key}: {json.dumps(res)}")
        del dec32
        torch.cuda.empty_cache()
    json.dump(out, open(OUT, "w"), indent=1)
    # ---------------------------------------------------------------- (2) graph == eager, (3) ms / frame
    pre, lane, dep, pdep, A = conv_inputs(CONVS[0])
    P = pre["P"]
    NG = 150

    def run(e, graphs, seed=777):
        e.start(pre, seed)
        wrote, ms, toks, codes, pose = 0, [], [], [], []
        for f in range(NG):
            T = P + f
            need = (f + 1) * SPF + LOOK
            e.write_a(wrote, A[wrote:need]); wrote = need
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dec = e.step(T, graphs=graphs)
            tokB, _, _, _ = e.result(T, dec)
            ms.append(1e3 * (time.perf_counter() - t0))
            toks.append(int(tokB))
            if f >= CD:
                codes.append(list(e.last_codes)); pose.append(list(e.last_pose))
        return {"ms": ms, "tok": toks, "codes": codes, "pose": pose}
    r_e = run(eng, False)
    eng.capture()
    r_g = run(eng, True)
    ge = {"frames": NG, "lane_equal": r_e["tok"] == r_g["tok"], "codes_equal": r_e["codes"] == r_g["codes"], "pose_equal": r_e["pose"] == r_g["pose"]}
    r_g2 = run(eng, True)
    ge["graph_repeat_equal"] = r_g2["codes"] == r_g["codes"] and r_g2["pose"] == r_g["pose"]
    out["graph_vs_eager"] = ge
    out["ms_new"] = {"p50": float(np.percentile(r_g2["ms"][10:], 50)), "p95": float(np.percentile(r_g2["ms"][10:], 95))}
    LP.log(f"graph vs eager: {json.dumps(ge)} ms_new {out['ms_new']}")
    json.dump(out, open(OUT, "w"), indent=1)
    del eng
    torch.cuda.empty_cache()
    eo = LO.LivePPM(m, ldev, cfgs, devs, max_s=60.0)
    eo.capture()
    r_o = run(eo, True)
    out["ms_old"] = {"p50": float(np.percentile(r_o["ms"][10:], 50)), "p95": float(np.percentile(r_o["ms"][10:], 95))}
    out["old_vs_new_gen_codes_equal_frac"] = float(np.mean(np.asarray(r_o["codes"]) == np.asarray(r_g2["codes"])))
    LP.log(f"ms_old {out['ms_old']}")
    json.dump(out, open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
