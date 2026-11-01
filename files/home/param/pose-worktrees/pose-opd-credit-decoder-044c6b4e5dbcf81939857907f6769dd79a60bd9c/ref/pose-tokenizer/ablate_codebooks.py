#!/usr/bin/env python3
"""Part 1: render original vs decoded stills for consecutive (even/odd) poses,
  and quantify which original each decoded still matches.
Part 2: ablate active codebooks {4,6,8,10,12} and recompute the even-first vs
  odd-first consecutive contribution correlations (QUANTIZED path) to see if the
  pairing phenomenon depends on codebook count.

Temporarily sets config n_codebooks=12 to load all codebooks, then restores it.
"""
import sys, json; sys.path.insert(0, "/home/james/pose-tokenizer")
from pathlib import Path
import numpy as np, torch, cv2
from safetensors.torch import load_file
from pose_tokenizer import PoseTokenizer
from pose_tokenizer.data.kinematic import (
    NUM_JOINTS, PARENT_INDEX, TOPO_ORDER,
    FACE_CONNECTIONS, BODY_CONNECTIONS, LEFT_HAND_CONNECTIONS, RIGHT_HAND_CONNECTIONS)

MODEL_DIR = "/home/james/vansh-video-processing/pose-processing/pose-tokenizer-2x-12cb-lossshift1"
CFG = Path(MODEL_DIR) / "config.json"
SRC = "/mnt/som-fs/hands/pose-conditioning-keypoints/125194882901565440_00-00.00_to_00-15.03/125194882901565440_00-00.00_to_00-15.03_poses_chunk0000.safetensors"
OUT = Path("/home/james/pose-tokenizer/decoder_evidence")
W, H, PAD = 256, 480, 0.08
_CONN = FACE_CONNECTIONS + BODY_CONNECTIONS + LEFT_HAND_CONNECTIONS + RIGHT_HAND_CONNECTIONS
_JOINTS = sorted({i for p in _CONN for i in p})


def pos_off(off):
    out = np.empty_like(off)
    for j in TOPO_ORDER:
        p = PARENT_INDEX[j]
        out[..., j, :] = (0 if p == -1 else out[..., p, :]) + off[..., j, :]
    return out


def panel(feat_row, fitp, label):
    f = feat_row.reshape(NUM_JOINTS, 3).astype(np.float32)
    P = pos_off(f[None, :, :2])[0]
    sy, sx, oy, ox = fitp
    kn = np.stack([(P[:, 0]*sy+oy)/H, (P[:, 1]*sx+ox)/W], axis=-1)
    c = np.zeros((H, W, 3), np.uint8); h, w = c.shape[:2]
    yp, xp = kn[:, 0]*h, kn[:, 1]*w
    for i, j in _CONN:
        cv2.line(c, (int(xp[i]), int(yp[i])), (int(xp[j]), int(yp[j])), (0, 220, 0), 2, cv2.LINE_AA)
    for idx in _JOINTS:
        cv2.circle(c, (int(xp[idx]), int(yp[idx])), 3, (0, 220, 0), -1, cv2.LINE_AA)
    cv2.putText(c, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return c


def fit_params(P):
    y, x = P[..., 0], P[..., 1]
    sy = H*(1-2*PAD)/max(y.max()-y.min(), 1e-6); sx = W*(1-2*PAD)/max(x.max()-x.min(), 1e-6)
    return sy, sx, (H-sy*(y.min()+y.max()))/2, (W-sx*(x.min()+x.max()))/2


def pearson(a, b):
    a = a-a.mean(); b = b-b.mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))


def main():
    orig_cfg = json.loads(CFG.read_text())
    try:
        cfg = dict(orig_cfg); cfg["n_codebooks"] = 12
        CFG.write_text(json.dumps(cfg, indent=2))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = PoseTokenizer.from_pretrained(MODEL_DIR, device=dev)
        m = tok.model
        print("loaded codebooks:", len(m.quantizer.quantizers))

        o = load_file(SRC)["reconstruction"].numpy().astype("float32")
        T_in = o.shape[0]
        x0 = torch.from_numpy(o).float().to(dev)
        codes_full = tok.encode(x0)                      # list of 12, each (T_codes,)

        def decode_n(codes, n):
            m.set_active_codebooks(n)
            return tok.decode([c for c in codes[:n]]).cpu().numpy()

        # ---------- PART 1: original vs decoded strip (n=4), stills 20..27 ----------
        rec4 = decode_n(codes_full, 4).reshape(-1, NUM_JOINTS, 3)
        oo = o.reshape(-1, NUM_JOINTS, 3)
        s0, ns = 20, 8
        fitp = fit_params(pos_off(oo[s0:s0+ns, :, :2]))
        rows = []
        for tag, arr in [("ORIG", oo), ("DEC4", rec4)]:
            rows.append(np.concatenate(
                [panel(arr[s].reshape(-1), fitp, f"{tag} {s}{'(even)' if s%2==0 else '(odd)'}")
                 for s in range(s0, s0+ns)], axis=1))
        cv2.imwrite(str(OUT/"even_odd_orig_vs_decoded.png"),
                    cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
        print("wrote", OUT/"even_odd_orig_vs_decoded.png")

        # which original does each decoded still match? (steady token pairs)
        def off(a): return a[:, :2].reshape(-1)
        d_even_oe = d_even_oo = d_odd_oe = d_odd_oo = 0.0; nP = 0
        for k in range(5, T_in//2 - 2):
            e, od = 2*k, 2*k+1
            if od >= len(rec4): break
            de, do = off(rec4[e]), off(rec4[od]); oe, oodd = off(oo[e]), off(oo[od])
            d_even_oe += np.linalg.norm(de-oe); d_even_oo += np.linalg.norm(de-oodd)
            d_odd_oe += np.linalg.norm(do-oe); d_odd_oo += np.linalg.norm(do-oodd); nP += 1
        print(f"\n[match] avg ||decoded-original|| over {nP} steady token-pairs (n=4):")
        print(f"  decoded EVEN still vs orig-even={d_even_oe/nP:.3f}  vs orig-odd={d_even_oo/nP:.3f}")
        print(f"  decoded ODD  still vs orig-even={d_odd_oe/nP:.3f}  vs orig-odd={d_odd_oo/nP:.3f}")

        # ---------- PART 2: codebook ablation of even/odd correlations ----------
        # QUANTIZED forward-difference contribution matrix per n.
        EPS, K = 0.08, 6
        rng = np.random.default_rng(0)
        Ns = [4, 6, 8, 10, 12]
        base = {n: decode_n(codes_full, n) for n in Ns}
        T_out = base[4].shape[0]
        Macc = {n: np.zeros((T_out, T_in)) for n in Ns}
        for i in range(T_in):
            for _ in range(K):
                d = torch.from_numpy(rng.standard_normal(o.shape[1]).astype("float32")).to(dev); d /= d.norm()
                xp = x0.clone(); xp[i] = xp[i] + EPS*d
                cp = tok.encode(xp)
                for n in Ns:
                    r = decode_n(cp, n)
                    Macc[n][:, i] += np.linalg.norm(r - base[n], axis=1)
        print("\n[ablation] even-first (within-token) vs odd-first (across-token) consecutive correlation:")
        print(f"{'n_cb':>5} {'within':>8} {'across':>8} {'diff':>8}")
        WARM = 6
        for n in Ns:
            M = Macc[n]
            ev = [pearson(M[s], M[s+1]) for s in range(WARM, T_out-1) if s % 2 == 0]
            od = [pearson(M[s], M[s+1]) for s in range(WARM, T_out-1) if s % 2 == 1]
            print(f"{n:>5} {np.mean(ev):>8.3f} {np.mean(od):>8.3f} {np.mean(ev)-np.mean(od):>+8.3f}")
    finally:
        CFG.write_text(json.dumps(orig_cfg, indent=2))
        print(f"\nrestored config n_codebooks={orig_cfg.get('n_codebooks')}")


if __name__ == "__main__":
    main()
