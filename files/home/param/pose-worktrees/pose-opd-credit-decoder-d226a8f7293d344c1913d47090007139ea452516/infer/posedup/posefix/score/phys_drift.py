"""posefix 10-26: per-10 s drift of rendered-skeleton quality for the user's long call + real reference: mean joint confidence, share of
drawn-but-unconfident joints (0.2 < conf <= 0.3, red in the raster), share below the raster floor (conf <= 0.2), and phys 'any' fail at
conf_t 0.3 and 0.2 (as rendered). -> out/phys_drift.json"""
import json, os, sys
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import numpy as np
import score as S
RUNS = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/runs"
import importlib


def phys_mod(conf):
    os.environ["PHYS_CONF"] = str(conf)
    import phys
    return importlib.reload(phys)


def k55_run(rid):
    summ = json.load(open(f"{RUNS}/{rid}/summary.json"))
    N = int(summ.get("prefill_frames") or 0)
    ck, sk, _ = S.prep_codes(S._fix_codes(np.load(f"{RUNS}/{rid}/tokens.npz")["pose"]), N)
    kp, kp0 = S.to_keypoints(S.decode(ck), S.frame0_of(summ.get("person") or S.DEFAULT_PERSON))
    k0 = kp0.numpy()
    return kp.numpy()[:, S.P.SEL].astype(np.float64)[2 * sk:], max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3)


def bins(k55, sw, P3, P2, B=250):
    c = k55[..., 2]
    f3, f2 = P3.frames(k55, sw)[1].any(1), P2.frames(k55, sw)[1].any(1)
    out = []
    for i in range(0, len(c), B):
        cc = c[i:i + B]
        if len(cc) < B // 2:
            break
        out.append({"t0_s": i / 25, "conf_mean": round(float(cc.mean()), 3), "red_pct": round(100 * float(((cc > 0.2) & (cc <= 0.3)).mean()), 1),
                    "below_floor_pct": round(100 * float((cc <= 0.2).mean()), 1), "hand_conf": round(float(cc[:, 13:].mean()), 3),
                    "body_conf": round(float(cc[:, :13].mean()), 3),
                    "phys_any_c03_pct": round(100 * float(f3[i:i + B].mean()), 2), "phys_any_c02_pct": round(100 * float(f2[i:i + B].mean()), 2)})
    return out


def main():
    P3 = phys_mod(0.3); P3c = P3.calib()
    P2 = phys_mod(0.2); P2c = P2.calib()
    P3.calib = lambda c=P3c: c; P2.calib = lambda c=P2c: c
    res = {}
    for rid in ("20261026-205651-b62a", "20261026-205615-7b5a"):
        k, sw = k55_run(rid)
        res[rid] = bins(k, sw, P3, P2)
    z = np.load(os.path.join(H, "work", "phys_real_k55.npz"), allow_pickle=True)
    real = [s for s in z["sets"] if s[0] == "aloe"][:6]
    res["real_aloe_6tracks"] = [bins(k.astype(np.float64), sw, P3, P2)[:6] for _, _, k, sw in real]
    json.dump(res, open(os.path.join(H, "out", "phys_drift.json"), "w"), indent=1)
    for k_, v in res.items():
        print(k_, json.dumps(v)[:3000], flush=True)


if __name__ == "__main__":
    main()
