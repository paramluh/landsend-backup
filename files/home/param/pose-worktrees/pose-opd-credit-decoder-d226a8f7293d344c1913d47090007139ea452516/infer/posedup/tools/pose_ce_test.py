#!/usr/bin/env python3
"""aloe-posedup-live ENGINE CHECK (GPU; before the server loads): live_ppm_pose.LivePPM on a real training conversation of the pose-duplex FT
(voxsamples' convs/cN.npz: the training micro's system block, B lane / Mimi / pose at delay 2, the partner's raw-gain audio as A), every
frame TEACHER-FORCED (lane, Mimi and pose codes of B), A = the partner's audio at the same frame (= training):
  1. graphed vs eager step: pose cb0 logits / speech cb0 logits / text logits / hn per frame (max abs diff, first NP frames);
  2. teacher-forced CE vs the run's eval (step 475: pose cb0 1.07, pose light 2.94, speech cb0 1.03): pose cb0 CE, pose cb1-31 CE (fused
     pose decoder, slot 0 + D33 link), speech cb0 CE, joint text CE; alignment check: the same logits scored against the targets of frame
     f - 1 / f + 1: speech cb0 must be clearly lowest at f; pose cb0 must beat f + 1 (pose is STICKY: its cb0 repeats the previous code
     ~55 % of frames, so an honest model puts its change-frame mass on 'stay' and the f - 1 target scores lower by construction; the
     pose-input alignment is judged by the CE matching the run's eval instead).
usage: python tools/pose_ce_test.py <export dir> <out json> [conv npz] [frames] [side]"""
import json
import os
import sys
import time

H = "/data/home/amartyavarma/ppm/aloe-posedup-live"
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import live_ppm_pose as LP  # noqa: E402

CK, OUT = sys.argv[1], sys.argv[2]
CONV = sys.argv[3] if len(sys.argv) > 3 else "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c1.npz"
NF = int(sys.argv[4]) if len(sys.argv) > 4 else 500
K = int(sys.argv[5]) if len(sys.argv) > 5 else 0
NP = 40
SPF, CD, OFF = LP.SPF, LP.CD, LP.OFF


def ce(z, y):
    return float(torch.nn.functional.cross_entropy(z.float(), torch.as_tensor(y, device=z.device).view(-1), reduction="sum"))


@torch.no_grad()
def main():
    t0 = time.time()
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    m, ldev, cfgs, info = LP.build_model(CK, devs)
    eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=60.0)
    eng.capture()
    assert eng.pose and eng.bblink, info
    z = np.load(CONV)
    pre = {"P": int(z[f"pre_ids_{K}"].shape[1]), "ids": z[f"pre_ids_{K}"], "depth": z[f"pre_depth_{K}"], "present": z[f"pre_present_{K}"]}
    assert (z[f"pre_pdepth_{K}"] < 0).all(), "prompt positions carry pose?"
    P = pre["P"]
    lane, dep, pdep = z[f"laneB_{K}"], z[f"depB_{K}"], z[f"pdepB_{K}"]
    A = z[f"stem_{1 - K}"]
    nf = min(NF, len(lane))
    log = LP.log
    log(f"conv {CONV} side {K}: P {P}, frames {nf}, pose frames {(pdep[:nf, 0] >= 0).sum()}, speech frames {(dep[:nf, 0] >= 0).sum()}")

    def run(n, graphs, light=False):
        eng.start(pre, 1234)
        eng.write_a(0, A[:(n + 2) * SPF + 72])
        rec = {"zp0": [], "z0": [], "zt": [], "hn": [], "pl": []}
        for f in range(n):
            T = P + f
            frc = (int(lane[f]), dep[f] if dep[f, 0] >= 0 else None, pdep[f] if pdep[f, 0] >= 0 else None)
            eng.step(T, graphs=graphs, force=frc)
            eng.result(T, T >= P + CD)
            torch.cuda.synchronize(eng.d0)
            rec["zp0"].append(eng.zp0[0].clone()); rec["z0"].append(eng.z0[0].clone()); rec["zt"].append(eng.zt[0].clone())
            rec["hn"].append(eng.hn_p[0].float().clone())
            if light and pdep[f, 0] >= 0:
                with torch.cuda.device(eng.d0):
                    zb = m.decp_bb(eng.hn_p[0:1]) * float(m.decp_bb_alpha)   # 10-25: row 0 = cond (pose CFG engine: 2 rows)
                    loc = torch.as_tensor((pdep[f] - OFF)[None], device=eng.d0)
                    rec["pl"].append((f, eng.dec_p.teacher_logits(eng.hv_p[0:1], loc, zb)[0]))
        return rec

    res = {"ck": CK, "conv": CONV, "side": K, "P": P, "frames": nf, "info": {k: info[k] for k in ("step", "s", "s_src", "pose", "decp_bb", "decp_bb_alpha")}}
    # 1. graphed vs eager
    g = run(NP, True)
    e = run(NP, False)
    par = {}
    for k in ("zp0", "z0", "zt", "hn"):
        d = [float((a.float() - b.float()).abs().max()) for a, b in zip(g[k], e[k])]
        par[k] = {"max_abs_diff": max(d), "frames": len(d)}
    res["graph_vs_eager"] = par
    log(f"graph vs eager: {par}")
    # 2. teacher-forced CE
    r = run(nf, True, light=True)
    zp0 = torch.stack(r["zp0"]); z0 = torch.stack(r["z0"]); zt = torch.stack(r["zt"])
    out = {}
    for name, zz, tgt in (("pose_cb0", zp0, pdep[:nf, 0]), ("speech_cb0", z0, dep[:nf, 0])):
        for sh in (0, -1, 1):
            idx = [f for f in range(nf) if 0 <= f + sh < nf and tgt[f + sh] >= 0 and tgt[f] >= 0]
            y = np.asarray([tgt[f + sh] for f in idx])
            out[f"{name}_ce{'' if sh == 0 else f'_shift{sh:+d}'}"] = ce(zz[idx], y) / max(1, len(idx))
        idx = [f for f in range(nf) if tgt[f] >= 0]
        out[f"{name}_top1"] = float((zz[idx].argmax(-1).cpu().numpy() == tgt[idx]).mean())
        out[f"{name}_n"] = len(idx)
    idx = list(range(1, nf))
    out["text_ce_joint"] = ce(zt[idx], lane[idx]) / len(idx)
    if r["pl"]:
        tot, n, per = 0.0, 0, np.zeros(31)
        for f, zl in r["pl"]:
            y = torch.as_tensor(pdep[f, 1:] - OFF[1:], device=zl.device)
            c = torch.nn.functional.cross_entropy(zl.float(), y, reduction="none").cpu().numpy()
            per += c; tot += float(c.sum()); n += 31
        out["pose_light_ce"] = tot / n
        out["pose_light_ce_cb1"] = float(per[0] / len(r["pl"])); out["pose_light_ce_cb31"] = float(per[30] / len(r["pl"]))
    ref = {"pose_cb0_ce": 1.0695, "pose_light_ce": 2.9396, "speech_cb0_ce": 1.0326}
    res["ce"] = out
    res["eval_ref_step475"] = ref
    ok_par = par["zp0"]["max_abs_diff"] < 0.5 and par["hn"]["max_abs_diff"] < 0.5
    p0 = pdep[2:nf, 0]
    out["pose_cb0_stay_frac"] = float((p0[1:] == p0[:-1]).mean())
    ok_al = out["pose_cb0_ce"] < out["pose_cb0_ce_shift+1"] and out["speech_cb0_ce"] < min(out["speech_cb0_ce_shift-1"], out["speech_cb0_ce_shift+1"])
    ok_ce = out["pose_cb0_ce"] < 1.5 * ref["pose_cb0_ce"] and out.get("pose_light_ce", 99) < 1.5 * ref["pose_light_ce"] and out["speech_cb0_ce"] < 2.0 * ref["speech_cb0_ce"]
    res["PASS"] = {"graph_vs_eager": ok_par, "alignment": ok_al, "ce_near_eval": ok_ce, "all": ok_par and ok_al and ok_ce}
    res["wall_s"] = round(time.time() - t0, 1)
    log(f"CE {json.dumps(out)}")
    log(f"PRETEST {'PASS' if res['PASS']['all'] else 'FAIL'} {res['PASS']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
