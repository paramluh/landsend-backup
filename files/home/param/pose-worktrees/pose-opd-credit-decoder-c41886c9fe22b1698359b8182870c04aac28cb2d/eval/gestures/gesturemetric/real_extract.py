"""gesturemetric: extract strokes + own / other-speaker words for real Aloe held-out tracks -> out/real/<conv>__<side>.npz.
Tracks = POSE_INDEX split 'eval' c00 convs with both sides ok (run_baselines.aloe_tracks order), first MAX_TOK tokens, person 01 frame
(= the scorer's convention; stroke detection is shoulder-width normalised so the frame choice only rescales px).
usage: real_extract.py [N_CONVS] [MAX_TOK] [SHARD/NSHARD]"""
import json
import os
import sys
import time

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import gm  # noqa: E402
import torch  # noqa: E402
import score as S  # noqa: E402

ALOE = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
POSE_INDEX = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/data/POSE_INDEX.json"
OUT = os.path.join(H, "out", "real")


def eval_convs():
    pix = json.load(open(POSE_INDEX))["rows"]
    ev = {}
    for d in pix.values():
        for v in d.values():
            if v["split"] == "eval" and v["row_key"].endswith("/c00"):
                sides = v["sides"]
                if all(sides.get(s, {}).get("ok") and sides[s]["T"] >= 1500 for s in ("assistant", "user")):
                    ev[v["conv"]] = v
    return [ev[c] for c in sorted(ev)]


def words_of(v):
    p = os.path.join(ALOE, os.path.dirname(v["sides"]["assistant"]["path"]), "w00.words.json")
    ws = json.load(open(p))["words"]
    by = {}
    for w in ws:
        by.setdefault(w.get("_separated_speaker"), []).append((float(w["start"]), float(w["end"]), w["text"]))
    return by


def main():
    n_conv = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    max_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 7500
    sh, nsh = (map(int, sys.argv[3].split("/")) if len(sys.argv) > 3 else (0, 1))
    torch.set_grad_enabled(False)
    os.makedirs(OUT, exist_ok=True)
    convs = eval_convs()[:n_conv]
    for ci, v in enumerate(convs):
        if ci % nsh != sh:
            continue
        by = words_of(v)
        for side, other in (("assistant", "user"), ("user", "assistant")):
            name = v["conv"].replace("/", "_") + "__" + side
            fo = os.path.join(OUT, name + ".npz")
            if os.path.exists(fo):
                continue
            t0 = time.time()
            s = v["sides"][side]
            codes = np.load(os.path.join(ALOE, s["path"]))
            codes = S._fix_codes(codes)
            k55, sw = gm.k55_of(codes, S.DEFAULT_PERSON, max_tok)
            ws = gm.wrist_speed(k55, sw)
            st, sv = gm.strokes(ws)
            L = k55.shape[0] / gm.FPS
            own = [w for w in by.get(s["spk"], []) if w[0] < L]
            oth = [w for w in by.get(v["sides"][other]["spk"], []) if w[0] < L]
            np.savez(fo + ".tmp.npz", st=st, sv=sv, ws=ws.astype(np.float32), L=L, sw=sw,
                     own=np.array(own, dtype=object), oth=np.array(oth, dtype=object), conv=v["conv"], side=side, spk=s["spk"])
            os.replace(fo + ".tmp.npz", fo)
            print(f"{name} L={L:.0f}s strokes={len(st)} ({60 * len(st) / L:.1f}/min) own words={len(own)} other={len(oth)} "
                  f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
