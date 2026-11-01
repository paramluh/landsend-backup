"""gesturemetric (2): score GENERATED pose with the CPU-mini judge (train2.py ckpt) -> judge/score2_<tag>.json.
Unit = 4 s window (50 steps), hop 12 steps (~1 s), measured part only (start >= prefill), >= 3 own words. A "source" = one 20 s cut:
a genweb run is one source, a live call / real eval track is cut into 20 s sources. For each target window: 15 distractor speech windows
= windows of the same source starting >= 4 s away, topped up with windows of OTHER sources in the same pool (pool = same real track /
same live call / same model + hidden flag + person for playground runs = constant speaker identity). top-1@16: does the pose window pick
its own speech? (chance 6.25 %). REAL ceiling = the identical protocol on the held-out eval tracks (pool = the track).
normalised meaning = (acc - 1/16) / (acc_real - 1/16): 0 = speech-unrelated (flailing), 1 = real-like.
Also: shuffled generated (pose of one source vs the speech windows of another source in its pool) = chance check; words-only / audio-only.
usage: score2.py [TAG] [--feat-dir DIR (default: the ckpt's feat dir, else feat)] [--pose-keep 5-10,13,14 (default: the ckpt's, else all)]; env GM_DEVICE=cuda runs decode / Qwen / judge on the GPU.
Generated-track features are cached in judge/<feat dir>/gen_cache.pt."""
import glob
import json
import os
import sys
import time

J = os.path.dirname(os.path.abspath(__file__))
H = os.path.dirname(J)
sys.path.insert(0, J)
sys.path.insert(0, H)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gm  # noqa: E402
import prep2  # noqa: E402
import train2 as T2  # noqa: E402
import score as S  # noqa: E402

RUNS = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/runs"
CUT, HOP = 250, 12            # 20 s sources, 1 s hop


def local_ids(m):
    m = np.asarray(m).astype(np.int64)
    if m.shape[0] == 32 and m.shape[1] != 32:
        m = m.T
    if (m >= 2048).any():
        m = np.where(m >= 0, m - np.arange(m.shape[1])[None] * 2048, 0)
    return np.clip(m, 0, 2047)


def gen_track(z):
    """one out/gen npz -> track dict like train2.load_split (pose, w_step, w_vec, mimi, T) + lo_step, or None."""
    src, run = str(z["src"]), str(z["run"])
    if src == "genweb-run":
        tk = np.load(os.path.join(RUNS, run, "tokens.npz"))
        pose, mimi = tk["pose"], tk["mimi"]
    else:
        sz = np.load(os.path.join(src, "session.npz"))
        pose, mimi = np.asarray(sz["pose"]).reshape(-1, 32), z["mimi_b"]
    codes = S._fix_codes(pose)
    if (codes < 0).any() or len(codes) < 60:
        return None
    k55, sw = gm.k55_of(codes, str(z["person"]), None, prep2.DEV)
    pf = prep2.pose_feats(k55, sw).astype(np.float32)
    T = min(pf.shape[0] // 2, len(mimi))
    own = [tuple(x) for x in z["own"].tolist()] if z["own"].size else []
    own = [(float(a), float(b), str(w)) for a, b, w in own if float(a) < T * 0.08]
    wv = prep2.word_vecs([w for _, _, w in own]).astype(np.float32)
    return {"name": run, "conv": run, "pose": pf[:2 * T], "w_step": np.array([int(a / 0.08) for a, _, _ in own], np.int64),
            "w_vec": wv, "mimi": local_ids(mimi)[:T, :8], "T": T, "lo_step": int(np.ceil(float(z["lo_s"]) / 0.08 - 1e-6))}


def windows_of(t, lo, hi):
    v = set(t["valid"].tolist())
    return [s for s in range(lo, hi - T2.WIN + 1, HOP) if s in v]


def sources(tracks, pool_of):
    """-> list of (track, window starts, pool key): 20 s cuts from lo_step."""
    out = []
    for t in tracks:
        lo = t.get("lo_step", 0)
        while lo + T2.WIN <= t["T"]:
            hi = t["T"] if t["T"] - lo < 1.5 * CUT else lo + CUT          # a short tail joins the last cut
            ws = windows_of(t, lo, hi)
            if ws:
                out.append((t, ws, pool_of(t)))
            if hi >= t["T"]:
                break
            lo = hi
    return out


@torch.no_grad()
def embed(model, srcs):
    E = []
    for t, ws, _ in srcs:
        p, w, m = T2.batch(None, [(t, ws)])
        E.append({"p": model.enc_pose(p), "s": model.enc_speech(w, m), "sw": model.enc_speech(w, m, True, False),
                  "sa": model.enc_speech(w, m, False, True)})
    return E


def retrieval(srcs, E, key="s", shuffled=False, seed=0):
    """top-1@16 over all target windows."""
    rng = np.random.default_rng(seed)
    pools = {}
    for i, (_, _, pk) in enumerate(srcs):
        pools.setdefault(pk, []).append(i)
    hits, n = 0, 0
    for i, (t, ws, pk) in enumerate(srcs):
        others = [j for j in pools[pk] if j != i]
        if shuffled:
            if not others:
                continue
            j = others[rng.integers(len(others))]
            k = min(len(ws), len(srcs[j][1]))
            P, Sp = E[i]["p"][:k], E[j][key][:k]
            wsS = srcs[j][1][:k]
        else:
            P, Sp, wsS = E[i]["p"], E[i][key], ws
        for a in range(len(P)):
            own = [b for b in range(len(wsS)) if abs(wsS[b] - wsS[a]) >= T2.MIN_GAP] if not shuffled else \
                  [b for b in range(len(wsS)) if b != a and abs(wsS[b] - wsS[a]) >= T2.MIN_GAP]
            cand = [Sp[b] for b in own[:15]]
            pool_e = [(j, b) for j in others for b in range(len(E[j][key]))] if len(cand) < 15 else []
            if len(cand) < 15:
                if len(pool_e) < 15 - len(cand):
                    continue
                for q in rng.choice(len(pool_e), 15 - len(cand), replace=False):
                    j, b = pool_e[q]
                    cand.append(E[j][key][b])
            C = torch.stack([Sp[a]] + cand)
            hits += int(torch.argmax(C @ P[a]) == 0)
            n += 1
    return (round(100 * hits / n, 2) if n else None), n


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "base"
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    ck = torch.load(os.path.join(J, f"ckpt_{tag}.pt"), weights_only=False, map_location=T2.DEV)
    T2.FEAT = sys.argv[sys.argv.index("--feat-dir") + 1] if "--feat-dir" in sys.argv else ck.get("feat", "feat")
    T2.set_arch(ck.get("arch", {}))
    T2.set_pose_keep(sys.argv[sys.argv.index("--pose-keep") + 1] if "--pose-keep" in sys.argv else ck.get("pose_keep"))
    model = T2.Judge().to(T2.DEV); model.load_state_dict(ck["state"]); model.eval()
    print(f"tag {tag} feat {T2.FEAT} arch {T2.arch()} device {T2.DEV} pose_keep {T2.POSE_KEEP or 'all'}", flush=True)
    mu, P = ck["pca_mu"], ck["pca_P"]
    res = {"tag": tag, "pose_keep": T2.POSE_KEEP}
    EV = T2.load_split("eval"); T2.prepare(EV, mu, P)
    rs = sources(EV, lambda t: t["name"])
    RE_ = embed(model, rs)
    res["real"] = {k: retrieval(rs, RE_, k) for k in ("s", "sw", "sa")}
    res["real_shuffled"] = retrieval(rs, RE_, "s", shuffled=True)
    print("real", res["real"], "shuffled", res["real_shuffled"], f"{time.time() - t0:.0f}s", flush=True)
    cache = os.path.join(J, T2.FEAT, "gen_cache.pt")
    G = torch.load(cache, weights_only=False) if os.path.exists(cache) else []
    for f in [] if G else sorted(glob.glob(os.path.join(H, "out", "gen", "*.npz"))):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f, allow_pickle=True)
        try:
            t = gen_track(z)
        except Exception as e:  # noqa: BLE001
            print("SKIP", os.path.basename(f), type(e).__name__, e, flush=True)
            continue
        if t is None:
            continue
        hid = bool(z["hidden"])
        t["group"] = f"{z['kind']}|{str(z['model'])[:40]}|{'hidden' if hid else 'user'}"
        t["pool"] = t["group"] + "|" + (str(z["run"]) if str(z["kind"]) == "session" else str(z["person"]))
        G.append(t)
    if not os.path.exists(cache):
        torch.save(G, cache + ".tmp"); os.replace(cache + ".tmp", cache)
    T2.prepare(G, mu, P)
    print(f"generated sources {len(G)} {time.time() - t0:.0f}s", flush=True)
    acc_real = res["real"]["s"][0]
    groups = {}
    for t in G:
        groups.setdefault(t["group"], []).append(t)
        groups.setdefault("ALL user-made" if t["group"].endswith("user") else "ALL hidden", []).append(t)
    res["gen"] = {}
    for g, ts in sorted(groups.items()):
        gs = sources(ts, lambda t: t["pool"])
        if not gs:
            continue
        E = embed(model, gs)
        a, n = retrieval(gs, E, "s")
        r = {"windows": n, "acc": a, "words_only": retrieval(gs, E, "sw")[0], "audio_only": retrieval(gs, E, "sa")[0],
             "shuffled": retrieval(gs, E, "s", shuffled=True)[0]}
        r["meaning_norm"] = None if a is None else round((a - 6.25) / (acc_real - 6.25), 3)
        res["gen"][g] = r
        print(g, r, flush=True)
    res["wall_s"] = round(time.time() - t0)
    so = os.path.join(J, f"score2_{tag}.json")
    json.dump(res, open(so + ".tmp", "w"), indent=1)
    os.replace(so + ".tmp", so)


if __name__ == "__main__":
    main()
