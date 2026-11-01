#!/usr/bin/env python3
"""posefix 10-26: light-decoder precision check for the MONOPLEX playground engine (aloe-pose-monoplex-infer fast_ppm; staged trainer-exact
copy in aloe-posedup-live/stage-exact/mono) on the pose monoplex export-357. = test_light.py's part (1) for FastPPM: held-out convs' B
track (lane + Mimi + pose teacher-forced per step, channel A empty = monoplex rows, the conv's own B voice clip as system block, no
persona), the decoders' inputs (hv, zb) recorded; per decoder REF (trainer eval path on an fp32 copy under bf16 autocast) vs NEW (staged
FusedDec) vs OLD (the deployed fast_ppm.FusedDec), + cb0 CE. Then graph == eager (free generation, same seed) and ms / frame NEW vs OLD.
usage: python test_light_mono.py <ck> <out json> <frames per conv> <conv npz> [...]"""
import copy
import importlib.util
import json
import os
import sys
import time

MI = "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer"
NEWDIR = "/data/home/amartyavarma/ppm/aloe-posedup-live/stage-exact/mono"
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
_sp = importlib.util.spec_from_file_location("fp_old", f"{MI}/ppm/fast_ppm.py")
FO_ = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(FO_)
assert FO_.FusedDec is not FP.FusedDec

CK, OUT, NF, CONVS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
CD, OFF, PAD, DEPTH, CBS, SPF = FP.CD, FP.OFF, FP.PAD, FP.DEPTH, FP.CBS, 1920
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim")
from mono import MONO_PARAMS  # noqa: E402


def ref_logits(dec32, hv, codes_local, zb, dev):
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


def conv_x(path, n, K=0):
    zc = np.load(path)
    d1 = zc[f"pre_depth_{K}"][1]
    clip = (d1[d1[:, 0] >= 0] - OFF).T.astype(np.int64)
    sp = MIF.system_block(clip, None)
    lane = np.full(n + CD, PAD, np.int64)
    x = MIF.assemble(sp, lane, np.zeros((DEPTH, 0), np.int64), np.zeros((DEPTH, 0), np.int64), n, 0)
    return x, zc[f"laneB_{K}"], zc[f"depB_{K}"], zc[f"pdepB_{K}"]


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
        dl, dp = eng.dl, eng.dp
        old = {"speech": FO_.FusedDec(m.dec_s, dl, 1), "pose": FO_.FusedDec(m.dec_p, dp, 1)}
        rec = {"speech": [], "pose": []}
        og_s, og_p = eng.dec_s.gen, eng.dec_p.gen

        def rec_s(hv, c0, *a, **k):
            rec["speech"].append([hv.clone(), None, None]); return og_s(hv, c0, *a, **k)

        def rec_p(hv, c0, *a, **k):
            zb = k.get("zb")
            rec["pose"].append([hv.clone(), None if zb is None else zb.clone(), None]); return og_p(hv, c0, *a, **k)
        eng.dec_s.gen, eng.dec_p.gen = rec_s, rec_p
        for ci, cp in enumerate(CONVS):
            x, lane, dep, pdep = conv_x(cp, NF)
            n0s, n0p = len(rec["speech"]), len(rec["pose"])
            eng.start([dict(x, a_mode="empty")], 4321 + ci, dict(MONO_PARAMS), f0=0)
            gen_f = []
            for f in range(NF + CD):
                frc = {"lane": [int(lane[f])]}
                if dep[f, 0] >= 0:
                    frc["codes"] = dep[f][None]
                if pdep[f, 0] >= 0:
                    frc["pose"] = pdep[f][None]
                eng.step(f, graphs=False, force=frc, cold_pad=False)
                if f >= CD:
                    gen_f.append(f)
            torch.cuda.synchronize()
            assert len(rec["speech"]) - n0s == len(gen_f) == len(rec["pose"]) - n0p, (len(rec["speech"]) - n0s, len(rec["pose"]) - n0p)
            for key, ca in (("speech", dep), ("pose", pdep)):
                for r, f in zip(rec[key][-len(gen_f):], gen_f):
                    r[2] = ca[f] if ca[f, 0] >= 0 else None
            print(f"[mono-light] conv {cp}: {len(gen_f)} frames", flush=True)
        eng.dec_s.gen, eng.dec_p.gen = og_s, og_p
        for key, dec, nd, dev, tw in (("speech", m.dec_s, eng.dec_s, dl, m.tower_s), ("pose", m.dec_p, eng.dec_p, dp, m.tower_p)):
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
                Z["c0"].append((hv @ tw.cb0_head.weight.t().to(hv.dtype)).float()[0])
                Z["y0"].append(ct[0, 0])
            zr, y = torch.stack(Z["ref"]), torch.stack(Z["y"])
            res = {"frames": len(Z["y"]), "ref_light_ce": float(F.cross_entropy(zr.reshape(-1, CBS), y.reshape(-1))),
                   "cb0_ce": float(F.cross_entropy(torch.stack(Z["c0"]), torch.stack(Z["y0"]))),
                   "new_vs_ref": metr(torch.stack(Z["new"]), zr, y), "old_vs_ref": metr(torch.stack(Z["old"]), zr, y)}
            out[key] = res
            print(f"[mono-light] {key}: {json.dumps(res)}", flush=True)
            del dec32
            torch.cuda.empty_cache()
        json.dump(out, open(OUT, "w"), indent=1)
        # graph == eager (free generation, same seed) and ms / frame
        x, lane, dep, pdep = conv_x(CONVS[0], NF)
        NG = min(150, NF)

        def run(e, graphs):
            e.start([dict(x, a_mode="empty")], 777, dict(MONO_PARAMS), f0=0)
            ms = []
            for f in range(NG + CD):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                e.step(f, graphs=graphs)
                torch.cuda.synchronize()
                ms.append(1e3 * (time.perf_counter() - t0))
            S = e.readback(0, int(x["P"]) + NG + CD)
            return {"ms": ms, "lane": S["lane_b"].tolist(), "dep": S["dep_b"].tolist(), "pdep": S["pdep"].tolist()}
        r_e = run(eng, False)
        eng.capture()
        r_g = run(eng, True)
        out["graph_vs_eager"] = {"frames": NG, "lane_equal": r_e["lane"] == r_g["lane"], "codes_equal": r_e["dep"] == r_g["dep"],
                                 "pose_equal": r_e["pdep"] == r_g["pdep"]}
        out["ms_new"] = {"p50": float(np.percentile(r_g["ms"][10:], 50)), "p95": float(np.percentile(r_g["ms"][10:], 95))}
        print(f"[mono-light] graph vs eager {json.dumps(out['graph_vs_eager'])} ms_new {out['ms_new']}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
        del eng
        torch.cuda.empty_cache()
        eo = FO_.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=True, max_s=max_s, mimi_stream=False, silence=sil)
        eo.capture()
        r_o = run(eo, True)
        out["ms_old"] = {"p50": float(np.percentile(r_o["ms"][10:], 50)), "p95": float(np.percentile(r_o["ms"][10:], 95))}
        print(f"[mono-light] ms_old {out['ms_old']}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
