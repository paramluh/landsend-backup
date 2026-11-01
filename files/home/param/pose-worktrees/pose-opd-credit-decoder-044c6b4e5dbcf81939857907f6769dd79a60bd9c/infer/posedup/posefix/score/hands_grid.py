import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import score as S, hands as HD
G = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/runs"
for r in sys.argv[1:]:
    rd = os.path.join(G, r)
    summ = json.load(open(os.path.join(rd, "summary.json")))
    N = int(summ.get("pose_prefill_frames") or 0)
    ck, sk, _ = S.prep_codes(S._fix_codes(np.load(os.path.join(rd, "tokens.npz"))["pose"]), N)
    kp, kp0 = S.to_keypoints(S.decode(ck), S.frame0_of("01_ab63d745"))
    k0 = kp0.numpy(); sw = max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3)
    hs, ws = HD.series(kp.numpy()[:, S.P.SEL].astype(np.float64)[2 * sk:], sw)
    out = {}
    for thr in (0.01, 0.015, 0.02):
        for mn in (0.16, 0.2, 0.3):
            b = HD.bursts((ws > thr).any(1), min_s=mn)
            out[f"{thr}/{mn}"] = [len(b), [round(a / 25 + 2.08, 1) for a, _ in b]]
    print(r, json.dumps(out))
