"""aloe-pose-mono-8020-sam: end-to-end data check with the trainer's own micro builder (data_aloe_pose.StepDS / eval_micros_v2).
  python micro_diff_check.py dump <env> <store dir> <code dir> <out.pkl>   -> 48 train micros (seeded (step, rank) picks incl. step 0 ranks 0..7)
                                                                           + 8 eval micros, as built for the trainer at world 64
  python micro_diff_check.py cmp <orig.pkl> <sam.pkl>                      -> per key: identical / differs (+ fraction of differing elements)
Expected: every key identical except the arrays that carry B's Mimi codes (speech frames and the voice-clip frames of the prefix)."""
import json
import os
import pickle
import sys

import numpy as np


def env_from(path):
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            out[k] = v
    return out


def dump(envf, store, code, out):
    os.environ.update(env_from(envf)); os.environ.update(ALOE_STORE_DIR=store, WORLD_SIZE="64", RANK="0", LOCAL_RANK="0")
    sys.path.insert(0, code); sys.path.insert(0, f"{code}/lib")
    import data_aloe_pose as A
    assert A.STORE_DIR == store and A.MONO
    cfg = A.aloe_cfg()
    p = A.make_plan_v2(64, cfg)
    rng = np.random.default_rng(20261026)
    picks = [(0, r) for r in range(8)] + [(int(rng.integers(p["steps"])), int(rng.integers(64))) for _ in range(40)]
    res = {"train": [], "eval": []}
    for t, r in picks:
        ds = A.StepDS(p["items"], [p["step_micros"][t][r]], cfg, r, p["items"][0])
        b = ds[0][0]
        res["train"].append(((t, r), {k: (np.asarray(v) if not isinstance(v, (str, dict, list)) else v) for k, v in b.items()}))
    ev = A.eval_micros_v2(p["eval_items"], p["eval_micros"][:8], cfg, 0)
    for i, b in enumerate(ev):
        res["eval"].append((i, {k: (np.asarray(v) if not isinstance(v, (str, dict, list)) else v) for k, v in b.items()}))
    pickle.dump(res, open(out, "wb"))
    print(f"dumped {len(res['train'])} train + {len(res['eval'])} eval micros from {store}", flush=True)


def same(a, b):
    if isinstance(a, np.ndarray) and a.dtype == object:
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (list, tuple)) and any(isinstance(x, np.ndarray) for x in a):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    return json.dumps(a, default=str, sort_keys=True) == json.dumps(b, default=str, sort_keys=True)


def cmp(pa, pb):
    A_, B_ = pickle.load(open(pa, "rb")), pickle.load(open(pb, "rb"))
    stat = {}
    for kind in ("train", "eval"):
        for (ka, ma), (kb, mb) in zip(A_[kind], B_[kind]):
            assert ka == kb and set(ma) == set(mb), (kind, ka, kb, sorted(set(ma) ^ set(mb)))
            for k in ma:
                s = stat.setdefault((kind, k), [0, 0, 0, 0])
                s[0] += 1
                if not same(ma[k], mb[k]):
                    s[1] += 1
                    x, y = np.asarray(ma[k]), np.asarray(mb[k])
                    if x.shape == y.shape and x.dtype != object:
                        s[2] += int((x != y).sum()); s[3] += int(x.size)
    out = {}
    for (kind, k), (n, nd, de, tot) in sorted(stat.items()):
        out[f"{kind}.{k}"] = {"micros": n, "micros_differing": nd, "elem_frac_differing": (round(de / tot, 4) if tot > 0 else None)}
        print(f"{kind:5s} {k:24s} {'IDENTICAL' if nd == 0 else 'DIFFERS'} in {nd}/{n} micros" + (f"; differing elements {de / tot:.4f}" if nd and tot else ""))
    json.dump(out, open(os.path.join(os.path.dirname(pb), "micro_diff.json"), "w"), indent=1)


if __name__ == "__main__":
    dump(*sys.argv[2:6]) if sys.argv[1] == "dump" else cmp(sys.argv[2], sys.argv[3])
