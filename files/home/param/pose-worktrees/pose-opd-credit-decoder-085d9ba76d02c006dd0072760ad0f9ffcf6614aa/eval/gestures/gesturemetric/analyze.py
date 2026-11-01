"""gesturemetric analysis: real Aloe (out/real) vs shuffled vs generated (out/gen) -> out/analysis.json (+ printed summary).
  real      own strokes vs own words (+ own loudness peaks when out/loud/<name>.npy exists)
  shuf_conv own strokes vs the SAME side's words from another conversation (rotated by one)
  listener  own strokes vs the OTHER speaker's words (real only)
Two scales: whole tracks (600 s) and CLIP windows (CLIP_S = 18 s, the measured part of a 20 s playground run after its ~2 s prefill),
so generated runs are compared against real windows of the same length.
usage: analyze.py [--lag-scan]"""
import glob
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import gm  # noqa: E402

OUT = os.path.join(H, "out")
CLIP_S = 18.0
EVENTS = ("words", "content", "loud")


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, d, "*.npz"))):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f, allow_pickle=True)
        r = {k: z[k] for k in z.files}
        r["name"] = os.path.basename(f)[:-4]
        r["own"] = [tuple(x) for x in r["own"].tolist()] if r["own"].size else []
        r["oth"] = [tuple(x) for x in r["oth"].tolist()] if "oth" in r and r["oth"].size else []
        r["loud"] = loud_of(r)
        r["L"] = float(r["L"])
        r["lo_s"] = float(r.get("lo_s", 0.0))
        rows.append(r)
    return rows


def loud_of(r):
    """-> (loudness-peak times, covered-until s) or None. Real: out/loud/<name>.npy (Mimi decode, first 240 s only); generated: the
    npz's 'wav' (b.wav, same clock as the pose: sample 0 = code frame 0)."""
    lf = os.path.join(OUT, "loud", r["name"] + ".npy")
    if os.path.exists(lf):
        env = np.load(lf).astype(np.float64)
        return gm.loud_events(env), len(env) / 100.0
    wav = str(r.get("wav", "")) if "wav" in r else ""
    if ".npz:" in wav:                                    # live session: streamed B audio array inside session.npz
        p, key = wav.rsplit(":", 1)
        if os.path.exists(p):
            x = np.load(p)[key].astype(np.float32).reshape(-1)
            env = gm.loudness_env(x, int(r.get("sr", 24000)))
            return gm.loud_events(env), len(env) / 100.0
        return None
    if wav and os.path.exists(wav):
        import soundfile as sf
        x, sr = sf.read(wav, dtype="float32", always_2d=True)
        env = gm.loudness_env(x.mean(1), sr)
        return gm.loud_events(env), len(env) / 100.0
    return None


def events(words, loud, kind):
    if kind == "loud":
        return None if loud is None else np.asarray(loud[0])
    return gm.word_events(words, kind)


def clip_scores(st, words, loud, lo, hi, seed=0):
    out = {}
    for k in EVENTS:
        ev = events(words, loud, k)
        if ev is None:
            continue
        h = min(hi, loud[1]) if k == "loud" else hi       # loudness only where the envelope exists
        if h - lo < 2 * gm.SHIFT_MIN + 1:
            continue
        out[k] = gm.timing(st, ev, lo, h, seed=seed)
    out["rows3"] = gm.word_stroke_table(st, words, lo, hi)
    out["rows_sub"] = gm.substance_rows(st, words, lo, hi)
    out["n_strokes"] = int(((st >= lo) & (st <= hi)).sum())
    out["spm"] = 60 * out["n_strokes"] / (hi - lo)
    out["speech_frac"] = speech_frac(words, lo, hi)
    return out


def speech_frac(words, lo, hi):
    tot = 0.0
    for t0, t1, _ in words:
        a, b = max(lo, t0), min(hi, t1)
        tot += max(0.0, b - a)
    return tot / (hi - lo)


def agg(clips, key):
    """mean (and p10 / p90, sem) over clips of timing_pp / fine_pp for event kind key; plus pooled hit / chance."""
    v = [c[key] for c in clips if key in c and c[key]["hit"] is not None]
    if not v:
        return None
    tp = np.array([x["timing_pp"] for x in v]); fp = np.array([x["fine_pp"] for x in v]); kp = np.array([x["kappa"] for x in v])
    n = np.array([x["n"] for x in v])
    return {"clips": len(v), "strokes": int(n.sum()),
            "kappa": round(float(kp.mean()), 3), "kappa_sem": round(float(kp.std() / np.sqrt(len(kp))), 3),
            "kappa_p10_p90": [round(float(x), 3) for x in np.percentile(kp, [10, 90])],
            "timing_pp": round(float(tp.mean()), 2), "timing_sem": round(float(tp.std() / np.sqrt(len(tp))), 2),
            "timing_p10_p90": [round(float(x), 1) for x in np.percentile(tp, [10, 90])],
            "fine_pp": round(float(fp.mean()), 2), "fine_sem": round(float(fp.std() / np.sqrt(len(fp))), 2),
            "hit": round(float(np.average([x["hit"] for x in v], weights=n)), 4),
            "chance": round(float(np.average([x["chance"] for x in v], weights=n)), 4),
            "local": round(float(np.average([x["local"] for x in v], weights=n)), 4)}


def summarise(clips):
    rows3 = [r for c in clips for r in c["rows3"]]
    return {"n": len(clips), "spm": round(float(np.mean([c["spm"] for c in clips])), 1),
            "speech_frac": round(float(np.mean([c["speech_frac"] for c in clips])), 3),
            **{k: agg(clips, k) for k in EVENTS}, "meaning": gm.lifts(rows3),
            "meaning_boot": boot_lift(clips),
            "substance": gm.substance_lift([r for c in clips for r in c["rows_sub"]]), "substance_boot": boot_sub(clips)}


def boot_sub(clips, B=300, seed=0):
    rng = np.random.default_rng(seed)
    if len(clips) < 5:
        return None
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(clips), len(clips))
        L = gm.substance_lift([r for i in idx for r in clips[i]["rows_sub"]])
        if L:
            vals.append(L["lift"])
    return [round(float(x), 3) for x in np.percentile(vals, [5, 95])] if vals else None


def boot_lift(clips, B=300, seed=0):
    """bootstrap over clips: 'any' lift p5 / p95."""
    rng = np.random.default_rng(seed)
    if len(clips) < 5:
        return None
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(clips), len(clips))
        L = gm.lifts([r for i in idx for r in clips[i]["rows3"]])
        if L.get("any", {}).get("lift") is not None:
            vals.append(L["any"]["lift"])
    return [round(float(x), 3) for x in np.percentile(vals, [5, 95])] if vals else None


def windows(r, clip=CLIP_S):
    lo = r["lo_s"]
    out = []
    while lo + clip <= r["L"]:
        out.append((lo, lo + clip)); lo += clip
    return out


def real_sets(R):
    res = {}
    n = len(R)
    whole = {"real": [], "shuf_conv": [], "listener": []}
    win = {"real": [], "shuf_conv": [], "listener": []}
    for i, r in enumerate(R):
        o = R[(i + 2) % n]            # +2: skip this conv's other side (sorted names: conv__assistant, conv__user)
        L = min(r["L"], o["L"])
        st = r["st"]
        whole["real"].append(clip_scores(st, r["own"], r["loud"], 0, r["L"], seed=i))
        whole["shuf_conv"].append(clip_scores(st, o["own"], o["loud"], 0, L, seed=i))
        whole["listener"].append(clip_scores(st, r["oth"], None, 0, r["L"], seed=i))
        for j, (a, b) in enumerate(windows(r)):
            if b > L:
                break
            win["real"].append(clip_scores(st, r["own"], r["loud"], a, b, seed=1000 * i + j))
            win["shuf_conv"].append(clip_scores(st, o["own"], o["loud"], a, b, seed=1000 * i + j))
            win["listener"].append(clip_scores(st, r["oth"], None, a, b, seed=1000 * i + j))
    for k in whole:
        res["whole_" + k] = summarise(whole[k])
        res["clip_" + k] = summarise(win[k])
    return res


def gen_sets(G):
    groups = {}
    skipped = []
    for i, r in enumerate(G):
        lo, hi = r["lo_s"], r["L"]
        if hi - lo < 2 * gm.SHIFT_MIN + 1:
            skipped.append((r["name"], round(hi - lo, 1)))
            continue
        # playground runs (~20 s after prefill) = one clip; long live calls -> CLIP_S windows, like the real clips
        spans = [(lo, hi)] if hi - lo <= 1.5 * CLIP_S else windows(r)
        g = f"{r.get('kind', 'gen')}|{str(r.get('model', ''))[:40]}|{'hidden' if bool(r.get('hidden', False)) else 'user'}"
        for j, (a, b) in enumerate(spans):
            c = clip_scores(r["st"], r["own"], r["loud"], a, b, seed=1000 * i + j)
            c["name"] = r["name"]
            groups.setdefault(g, []).append(c)
            groups.setdefault("ALL generated", []).append(c)
            groups.setdefault("ALL user-made" if not bool(r.get("hidden", False)) else "ALL hidden", []).append(c)
    print(f"generated skipped (< {2 * gm.SHIFT_MIN + 1:.0f} s after prefill): {len(skipped)} {skipped[:8]}", flush=True)
    return {g: summarise(v) for g, v in groups.items()}


def lag_scan(R):
    """pooled excess density of (stroke - event) over the circular-shift null, for own words / content onsets."""
    out = {}
    for kind in ("words", "content"):
        hr, hn = 0, 0
        rng = np.random.default_rng(0)
        for r in R:
            ev = gm.word_events(r["own"], kind)
            hr = hr + gm.lag_hist(r["st"], ev, 0, r["L"])
            for d in rng.uniform(gm.SHIFT_MIN, r["L"] - gm.SHIFT_MIN, 20):
                hn = hn + gm.lag_hist(r["st"], np.sort((ev + d) % r["L"]), 0, r["L"]) / 20
        centres = np.arange(-1.5, 1.5, 0.04) + 0.02
        ratio = hr / np.maximum(hn, 1e-9)
        out[kind] = {"centre_s": [round(float(c), 2) for c in centres], "ratio": [round(float(x), 3) for x in ratio]}
        k = int(np.argmax(np.convolve(ratio, np.ones(5) / 5, mode="same")))
        print(f"lag scan {kind}: peak ratio {ratio.max():.3f} at {centres[np.argmax(ratio)]:+.2f}s (smoothed peak {centres[k]:+.2f}s); "
              + " ".join(f"{c:+.2f}:{x:.2f}" for c, x in zip(centres[::3], ratio[::3])), flush=True)
    return out


def fmt(name, s):
    def t(k):
        a = s.get(k)
        return "-" if not a else (f"k {a['kappa']:+.3f}±{a['kappa_sem']:.3f} pp {a['timing_pp']:+.1f}±{a['timing_sem']:.1f} "
                                  f"(fine {a['fine_pp']:+.1f}±{a['fine_sem']:.1f}) hit {a['hit']:.3f}/ch {a['chance']:.3f}")
    m = s["meaning"].get("any", {})
    cats = " ".join(f"{c}:{s['meaning'][c]['lift']}" for c in ("deictic", "self", "size", "negation") if c in s["meaning"])
    return (f"{name:55s} n={s['n']:4d} spm={s['spm']:5.1f} sp={s['speech_frac']:.2f} | words {t('words')} | content {t('content')} | "
            f"loud {t('loud')} | lift any {m.get('lift')} (n {m.get('n')}) boot {s['meaning_boot']} [{cats}] | substance {(s['substance'] or {}).get('lift')} boot {s['substance_boot']}")


def main():
    R = load("real")
    G = load("gen") if os.path.isdir(os.path.join(OUT, "gen")) else []
    print(f"real tracks {len(R)} (loud {sum(r['loud'] is not None for r in R)}), generated {len(G)} (loud {sum(r['loud'] is not None for r in G)})", flush=True)
    res = {"params": {k: getattr(gm, k) for k in ("PEAK_THR", "PROM", "MIN_SEP_S", "MERGE_S", "W", "LAG", "K_SHIFT", "SHIFT_MIN",
                                                    "JIT", "PRE", "POST")}, "clip_s": CLIP_S}
    if "--lag-scan" in sys.argv:
        res["lag_scan"] = lag_scan(R)
    res["real"] = real_sets(R)
    for k, s in res["real"].items():
        print(fmt(k, s), flush=True)
    if G:
        res["gen"] = gen_sets(G)
        for k, s in sorted(res["gen"].items()):
            print(fmt(k, s), flush=True)
    fo = os.path.join(OUT, "analysis.json")                # read live by gm_run.ref(): atomic replace
    json.dump(res, open(fo + ".tmp", "w"), indent=1, default=float)
    os.replace(fo + ".tmp", fo)


if __name__ == "__main__":
    main()
