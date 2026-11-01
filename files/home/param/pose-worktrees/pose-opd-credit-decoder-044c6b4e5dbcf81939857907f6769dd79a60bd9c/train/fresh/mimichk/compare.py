# fresh (new-pose-llm-data, lufs23-local-v1) vs TikTok-pose-data Mimi codes of the SAME clip: per-codebook agreement, first 10 s vs after
import glob, pickle, json, numpy as np, collections
M = "/data/home/amartyavarma/ppm/code-811/fresh/mimichk"
fresh = [x for f in sorted(glob.glob(f"{M}/fresh-*.pkl")) for x in pickle.load(open(f, "rb"))]
tk = pickle.load(open(f"{M}/tiktok.pkl", "rb"))
byc = collections.defaultdict(list)
for r in tk: byc[r["clip_id"]].append(r)
pairs, nomatch = [], 0
for x in fresh:
    c = byc.get(x["clip_id"]) or []
    k = int(x["segment_index"])
    cand = [r for r in c if int(r.get("segment", -1) if r.get("segment") is not None else -1) in ((k, -1) if k == 0 else (k,))]
    if not cand: nomatch += 1; continue
    pairs.append((x, cand[0]))
print("fresh overlap segments", len(fresh), "tiktok rows", len(tk), "same-clip pairs", len(pairs), "no same-clip match", nomatch)
CB = (0, 1, 2, 4, 8, 16, 31)
acc = {(b, part): [0, 0] for b in CB for part in ("<10s", ">=10s", "all")}
shifts = collections.Counter(); lens = []
for x, r in pairs:
    a, b = x["mimi"], r["mimi"]
    best = None
    for s in range(-3, 4):                       # alignment: fresh[t] vs tiktok[t + s]
        lo, hi = max(0, -s), min(a.shape[1], b.shape[1] - s)
        if hi - lo < 10: continue
        m = (a[0, lo:hi] == b[0, lo + s:hi + s]).mean()
        if best is None or m > best[0]: best = (m, s, lo, hi)
    if best is None: continue
    _, s, lo, hi = best; shifts[s] += 1; lens.append((a.shape[1], b.shape[1]))
    t = np.arange(lo, hi)
    for cb in CB:
        eq = a[cb, lo:hi] == b[cb, lo + s:hi + s]
        for part, msk in (("<10s", t < 125), (">=10s", t >= 125), ("all", np.ones_like(t, bool))):
            acc[(cb, part)][0] += int(eq[msk].sum()); acc[(cb, part)][1] += int(msk.sum())
print("best shifts", dict(shifts), "len pairs (fresh, tiktok) sample", lens[:8])
for cb in CB:
    print(f"cb{cb:<2d} " + "  ".join(f"{p}: {acc[(cb, p)][0] / max(1, acc[(cb, p)][1]):.3f} (n={acc[(cb, p)][1]})" for p in ("<10s", ">=10s", "all")))
x, r = pairs[0]
print("example", x["clip_id"], x["segment_index"], "fresh T", x["mimi"].shape, "tiktok", r.get("kind"), r.get("segment"), r["mimi"].shape, r["file"].split("/")[-2:])
print(" fresh cb0[:20]", x["mimi"][0, :20].tolist()); print(" tikt  cb0[:20]", r["mimi"][0, :20].tolist())
print(" fresh words", (x["words"] or [])[:12]); print(" tikt  words", (r["words"] or [])[:12])
print(" fresh segmeta", str(x["segmeta"])[:600])
