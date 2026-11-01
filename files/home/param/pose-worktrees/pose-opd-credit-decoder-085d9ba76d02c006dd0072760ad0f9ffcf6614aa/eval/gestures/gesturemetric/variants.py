"""gesturemetric variants on real / shuffled / generated (re-detects strokes from the saved wrist speed 'ws'):
  A  stroke threshold x hit window sweep for (1) timing on 18 s clips: real vs shuf_conv separation (Cohen d per clip) and kappa
     = (hit - chance) / (1 - chance) (speech-density-free).
  B  (3) cheap meaning: the 4 fixed categories, plus a DATA-DRIVEN lexicon: per-word stroke lift learned on half the real convs (A),
     tested on the other half (B) (+ shuffled + generated with the lexicon learned on all real) -> out/variants.json"""
import collections
import json
import os
import sys
import zlib

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import gm  # noqa: E402
import analyze as A  # noqa: E402

THRS = (0.015, 0.025, 0.04)
WINS = (0.1, 0.2, 0.3)


def restroke(r, thr):
    old = gm.PEAK_THR, gm.PROM
    gm.PEAK_THR, gm.PROM = thr, thr / 2
    st, _ = gm.strokes(r["ws"].astype(np.float64))
    gm.PEAK_THR, gm.PROM = old
    return st


def clip_timing(st, words, lo, hi, w, seed, kind="words"):
    t = gm.timing(st, gm.word_events(words, kind), lo, hi, w=w, k=100, seed=seed)
    if t["hit"] is None:
        return None
    return t


def stats(v, key):
    a = np.array([x[key] for x in v if x is not None])
    return a


def sweep(R, G):
    out = {}
    n = len(R)
    for thr in THRS:
        sts = [restroke(r, thr) for r in R]
        gsts = [restroke(r, thr) for r in G]
        for w in WINS:
            for kind in ("words", "content"):
                real, shuf, gen = [], [], []
                for i, r in enumerate(R):
                    o = R[(i + 2) % n]
                    L = min(r["L"], o["L"])
                    for j, (a, b) in enumerate(A.windows(r)):
                        if b > L:
                            break
                        real.append(clip_timing(sts[i], r["own"], a, b, w, 1000 * i + j, kind))
                        shuf.append(clip_timing(sts[i], o["own"], a, b, w, 1000 * i + j, kind))
                for i, r in enumerate(G):
                    if r["L"] - r["lo_s"] >= 2 * gm.SHIFT_MIN + 1 and not bool(r.get("hidden", False)):
                        gen.append(clip_timing(gsts[i], r["own"], r["lo_s"], r["L"], w, i, kind))
                row = {}
                for nm, v in (("real", real), ("shuf", shuf), ("gen_user", gen)):
                    for key in ("timing_pp", "fine_pp", "kappa"):
                        a = stats(v, key)
                        row[f"{nm}_{key}"] = [round(float(a.mean()), 3), round(float(a.std()), 3), len(a)] if len(a) else None
                for key in ("timing_pp", "kappa", "fine_pp"):
                    a, b = stats(real, key), stats(shuf, key)
                    row[f"d_{key}"] = round(float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)), 3)
                    # per-run AUC: P(real clip > shuffled clip)
                    row[f"auc_{key}"] = round(float((a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean()), 3)
                out[f"thr{thr}_w{w}_{kind}"] = row
                print(f"thr {thr} W {w} {kind:7s} | real k {row['real_kappa'][0]:+.3f} pp {row['real_timing_pp'][0]:+.1f} fine "
                      f"{row['real_fine_pp'][0]:+.1f} | shuf k {row['shuf_kappa'][0]:+.3f} | gen k "
                      f"{row['gen_user_kappa'][0] if row['gen_user_kappa'] else None} pp {row['gen_user_timing_pp'][0] if row['gen_user_timing_pp'] else None} "
                      f"fine {row['gen_user_fine_pp'][0] if row['gen_user_fine_pp'] else None} | d kappa {row['d_kappa']} AUC {row['auc_kappa']} "
                      f"d pp {row['d_timing_pp']} AUC {row['auc_timing_pp']}", flush=True)
    return out


def word_rows(st, words, lo, hi):
    """(normalised word, has stroke in [t0 - PRE, t0 + POST]) for content words."""
    out = []
    st = np.sort(st)
    for t0, t1, w in words:
        if t0 < lo + gm.PRE or t0 > hi - gm.POST:
            continue
        nw = gm.norm_word(w)
        if not nw:
            continue
        a, b = np.searchsorted(st, [t0 - gm.PRE, t0 + gm.POST])
        out.append((nw, bool(b > a)))
    return out


def learn_lex(rows, min_n=15, top=60):
    c = collections.defaultdict(lambda: [0, 0])
    for w, s in rows:
        c[w][0] += 1; c[w][1] += s
    base = sum(s for _, s in rows) / max(len(rows), 1)
    sc = {w: (s + 2 * base) / (n + 2) for w, (n, s) in c.items() if n >= min_n}   # shrunk rate
    order = sorted(sc, key=sc.get)
    return set(order[-top:]), set(order[:top]), base


def lex_lift(rows, hi_set, lo_set):
    h = [s for w, s in rows if w in hi_set]
    l = [s for w, s in rows if w in lo_set]
    if not h or not l or np.mean(l) == 0:
        return None
    return {"n_hi": len(h), "n_lo": len(l), "rate_hi": round(float(np.mean(h)), 4), "rate_lo": round(float(np.mean(l)), 4),
            "lift": round(float(np.mean(h) / np.mean(l)), 3)}


def boot(groups, hi_set, lo_set, B=300):
    rng = np.random.default_rng(0)
    v = []
    for _ in range(B):
        idx = rng.integers(0, len(groups), len(groups))
        r = lex_lift([x for i in idx for x in groups[i]], hi_set, lo_set)
        if r:
            v.append(r["lift"])
    return [round(float(x), 3) for x in np.percentile(v, [5, 95])] if v else None


def lexicon(R, G):
    st = {r["name"]: r["st"] for r in R}
    half = {r["name"]: zlib.crc32(str(r["conv"]).encode()) % 2 for r in R}
    n = len(R)
    real = {r["name"]: word_rows(st[r["name"]], r["own"], 0, r["L"]) for r in R}
    shuf = {r["name"]: word_rows(st[r["name"]], R[(i + 2) % n]["own"], 0, min(r["L"], R[(i + 2) % n]["L"])) for i, r in enumerate(R)}
    out = {}
    for test in (0, 1):
        tr = [x for k, v in real.items() if half[k] != test for x in v]
        hs, ls, base = learn_lex(tr)
        te = [v for k, v in real.items() if half[k] == test]
        ts = [v for k, v in shuf.items() if half[k] == test]
        out[f"test_half{test}"] = {"lex_hi": sorted(hs)[:60], "lex_lo": sorted(ls)[:60], "train_base": round(base, 4),
                                   "real": lex_lift([x for v in te for x in v], hs, ls), "real_boot": boot(te, hs, ls),
                                   "shuf": lex_lift([x for v in ts for x in v], hs, ls), "shuf_boot": boot(ts, hs, ls)}
        print(f"lexicon test half {test}: real {out[f'test_half{test}']['real']} boot {out[f'test_half{test}']['real_boot']} | "
              f"shuf {out[f'test_half{test}']['shuf']} boot {out[f'test_half{test}']['shuf_boot']}", flush=True)
    hs, ls, base = learn_lex([x for v in real.values() for x in v])
    gr = [word_rows(r["st"], r["own"], r["lo_s"], r["L"]) for r in G if not bool(r.get("hidden", False))]
    out["all_real_lex"] = {"hi": sorted(hs), "lo": sorted(ls)}
    out["gen_user"] = {"lift": lex_lift([x for v in gr for x in v], hs, ls), "boot": boot(gr, hs, ls) if len(gr) >= 5 else None}
    print(f"lexicon (all real) on generated user runs: {out['gen_user']}", flush=True)
    print("hi:", " ".join(sorted(hs)[:60]), "\nlo:", " ".join(sorted(ls)[:60]), flush=True)
    return out


def main():
    R = A.load("real")
    G = A.load("gen") if os.path.isdir(os.path.join(A.OUT, "gen")) else []
    G = [g for g in G if "ws" in g]
    print(f"real {len(R)} gen {len(G)}", flush=True)
    res = {"sweep": sweep(R, G), "lexicon": lexicon(R, G)}
    json.dump(res, open(os.path.join(A.OUT, "variants.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
