"""gesturemetric (2) EVAL-ONLY ablations of the FULL judge (ckpt_full.pt, judge/feat_full/eval), no retraining.
Reuses train2.py verbatim (load_split / prepare / pose_window / evaluate with seed 123, n_groups 8 = the SAME held-out groups as
eval2_full.json "final"); each ablation only wraps train2.batch and edits the tensors it returns:
  E full as-is (sanity: must reproduce eval2_full.json full p2s ~52.85)
  A words-only, word timing destroyed: per window a random permutation of the 50 step rows of the word features
    (word-vector sums + onset count; same multiset of words), audio zeroed. 3 seeds, mean.
  B words-only bag-of-words: window word features summed and spread uniformly over the 50 steps (sum / 50 at every step), audio zeroed.
  C full, pose head points (COCO body 0-4: x, y, velocities, conf masks) zeroed.
  D full, ONLY head points kept (everything else in the pose features, incl. hand spreads, zeroed).
Metric: top-1@16 pose->speech (chance 6.25 %). -> judge/ablate_full.json (atomic) + one line in judge/STATUS_FULL.md.
CPU only: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 python ablate_full.py"""
import json
import os
import sys
import time

J = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, J)
os.environ["GM_DEVICE"] = "cpu"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import train2 as T2  # noqa: E402

SEEDS_A = (0, 1, 2)
REF_FULL = 52.85

# per-frame pose features (train2.pose_window): rel xy [0:30] (point-major), spreads [30:32], conf masks [32:47], vel xy [47:77];
# each 4 s step = 2 frames concatenated -> 154. Head = points 0-4 (nose, eyes, ears).
_h = list(range(0, 10)) + list(range(32, 37)) + list(range(47, 57))
HEAD = np.array(_h + [i + 77 for i in _h])
NOT_HEAD = np.setdiff1d(np.arange(154), HEAD)

_ORIG_BATCH = T2.batch


def patched(fn):
    def b(*a, **k):
        p, w, m = _ORIG_BATCH(*a, **k)
        return fn(p, w, m)
    return b


def run(name, fn, mode):
    T2.batch = patched(fn) if fn is not None else _ORIG_BATCH
    try:
        r = T2.evaluate(MODEL, EV, n_groups=8, modes=(mode,), extra=False)[mode]
    finally:
        T2.batch = _ORIG_BATCH
    print(f"{name}: {r} {time.time() - t0:.0f}s", flush=True)
    return r


def perm_steps(seed):
    rng = np.random.default_rng(seed)

    def fn(p, w, m):
        idx = torch.from_numpy(np.stack([rng.permutation(T2.WIN) for _ in range(w.shape[0])]))
        return p, torch.gather(w, 1, idx[..., None].expand_as(w)), m
    return fn


def bow(p, w, m):
    return p, (w.sum(1, keepdim=True) / T2.WIN).expand_as(w).contiguous(), m


def zero_cols(cols):
    def fn(p, w, m):
        p = p.clone()
        p[..., torch.from_numpy(cols)] = 0
        return p, w, m
    return fn


if __name__ == "__main__":
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    ck = torch.load(os.path.join(J, "ckpt_full.pt"), map_location="cpu", weights_only=False)
    T2.FEAT = ck["feat"]
    T2.set_arch(ck["arch"])
    MODEL = T2.Judge()
    MODEL.load_state_dict(ck["state"])
    MODEL.eval()
    EV = T2.load_split("eval")
    T2.prepare(EV, ck["pca_mu"], ck["pca_P"])
    print(f"feat {T2.FEAT} arch {T2.arch()} eval sides {len(EV)} load {time.time() - t0:.0f}s", flush=True)

    res = {}
    res["E_full"] = run("E full as-is", None, "full")
    if abs(res["E_full"]["p2s"] - REF_FULL) > 0.5:
        print(f"SANITY FAIL: full p2s {res['E_full']['p2s']} != {REF_FULL}", flush=True)
        sys.exit(2)
    res["words_ref"] = run("words-only as-is", None, "words")
    a = [run(f"A perm seed {s}", perm_steps(s), "words") for s in SEEDS_A]
    res["A_words_perm"] = {"p2s": round(float(np.mean([x["p2s"] for x in a])), 2), "s2p": round(float(np.mean([x["s2p"] for x in a])), 2),
                           "p2s_seeds": [float(x["p2s"]) for x in a], "groups": a[0]["groups"]}
    res["B_words_bow"] = run("B bag-of-words", bow, "words")
    res["C_full_no_head"] = run("C no head", zero_cols(HEAD), "full")
    res["D_full_head_only"] = run("D head only", zero_cols(NOT_HEAD), "full")
    out = {"ckpt": "ckpt_full.pt", "feat": T2.FEAT, "eval_sides": len(EV), "metric": "top-1@16 (chance 6.25), eval seed 123 n_groups 8",
           "ref_eval2_full": {"full": REF_FULL, "words": 49.24, "audio": 46.16}, "results": res, "wall_s": round(time.time() - t0)}
    fo = os.path.join(J, "ablate_full.json")
    json.dump(out, open(fo + ".tmp", "w"), indent=1)
    os.replace(fo + ".tmp", fo)
    line = (f"- {time.strftime('%m-%d %H:%M', time.gmtime())} UTC [ablate_full.py, CPU]: eval-only ablations top-1@16 p2s: "
            f"E full {res['E_full']['p2s']} (ref {REF_FULL}), words-only {res['words_ref']['p2s']}, "
            f"A words timing-permuted {res['A_words_perm']['p2s']} (seeds {res['A_words_perm']['p2s_seeds']}), "
            f"B words bag-of-words {res['B_words_bow']['p2s']}, C no-head {res['C_full_no_head']['p2s']}, "
            f"D head-only {res['D_full_head_only']['p2s']} -> ablate_full.json\n")
    with open(os.path.join(J, "STATUS_FULL.md"), "a") as f:
        f.write(line)
    print(line, flush=True)
