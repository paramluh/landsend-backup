"""aloe-v2vox-raw UNIT TEST (CPU): the accumulation stream feeds every logical rank EXACTLY 22420's micro.
For physical ranks of the 3-node layout (ALOE_LNODES 0,1,2;3,4,5;6,7) and the 2-node layout: AccDS element (step t, slot g) == VoxDS of
logical rank L_g (= 22420's per-rank stream, same item list from rank_items(plan, L, 64), same rank argument) at step t, every array
(ids, depth, present, pid, sel, y, word, pad, eob, nw, cb0, light, own, vin, wav) bit-identical, same dummy flag; pad slots are dummies
with zero counts; the union of logical ranks over physical ranks = 0..63 once. Steps checked: 0, 1 and the last (plan dummies).
usage: VOX_A_GAIN=raw ALOE_FLAC_ROOT=<A tracks> python -I test_accds.py"""
import sys

import numpy as np

R = "/data/home/amartyavarma/ppm/aloe-v2vox-raw"
sys.path.insert(0, f"{R}/code"); sys.path.insert(0, f"{R}/code/lib")
KEYS = ("ids", "depth", "present", "pid", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light", "own", "vin", "wav")


def lranks_of(lnodes, P, j):
    G = max(len(g) for g in lnodes)
    return [8 * l + j for l in lnodes[P]] + [None] * (G - len(lnodes[P]))


def main():
    import data_aloe_v2vox as A
    PL = A.make_plan(64)
    fb = PL["items"][0]
    cfg = A.aloe_cfg()
    n = 0
    for lay in ("0,1,2;3,4,5;6,7", "0,1,2,3;4,5,6,7"):
        LN = [[int(x) for x in g.split(",")] for g in lay.split(";")]
        allL = sorted(L for P in range(len(LN)) for j in range(8) for L in lranks_of(LN, P, j) if L is not None)
        assert allL == list(range(64)), lay
        for P, j in ((len(LN) - 1, 3), (0, 0)):
            lr = lranks_of(LN, P, j)
            ds = A.AccDS([A.rank_items(PL, L, 64)[0] if L is not None else None for L in lr], lr, cfg, fb, singles_at=0)
            G = len(lr)
            for t in (0, 1, PL["steps"] - 1):
                for g in range(G):
                    e = ds[t * G + g]
                    assert e["step"] == t and e["slot"] == g
                    L = lr[g]
                    if L is None:
                        assert e["padslot"] == 1 and e.get("dummy") == 1 and not np.asarray(e["own"]).any(), (lay, P, j, t, g)
                        print(f"PASS {lay} P{P} j{j} t{t} slot{g}: pad slot = dummy, zero counts", flush=True)
                        continue
                    ref = A.VoxDS(A.rank_items(PL, L, 64)[0], cfg, L, fb, singles_at=0)[t]
                    assert e["lrank"] == L and int(e.get("dummy", 0)) == int(ref.get("dummy", 0)) and e["item"] == ref["item"], (lay, t, g, L)
                    for k in KEYS:
                        a, b = np.asarray(e[k]), np.asarray(ref[k])
                        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), (lay, t, g, L, k)
                    assert e["aref"] == ref["aref"] and e["voice_clip"] == ref["voice_clip"] and e["voice_drawn"] == ref["voice_drawn"]
                    n += 1
                    print(f"PASS {lay} P{P} j{j} t{t} slot{g} = logical rank {L}: item {e['item']} dummy {int(e.get('dummy', 0))} bit-identical to its VoxDS micro", flush=True)
    print(f"ALL PASS ({n} micros compared)", flush=True)


if __name__ == "__main__":
    main()
