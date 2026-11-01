"""si20k-vox SI20K_A_GAIN=normal: fit the population of REAL user-mic levels and check the training draw (CPU, offline; the ONLY offline
numbers are the population mean and SD).
  REAL      Aloe handover s2h-m1 channel A (the data 'user' = SPEAKER_01 of first-chunk rows, the WLT FLAC at 0 dB from the shared copy
            aloe-v2vox-raw/flac, kind gcs only), N random rows: active-speech level (data_si20k_vox.active_level_db: user word onsets ->
            turns (gap <= 2 s) -> spans [first onset, last onset + 0.6 s] -> dBFS over the concatenated spans).
  SI20K     the same measure on the si20k user channel (raw TTS level) for M rows of a store set.
  DRAW      SI20K_A_GAIN=normal on those si20k rows (data_si20k_vox.gain_db, epoch 0): resulting level, gain, peak-limit rate.
  DEMO      cross-check: the live demo sessions' mic input (session.npz 'a', 24 kHz, browser AEC on, AGC off; no word timings -> energy VAD:
            80 ms frames within 35 dB of the session's 95th-percentile frame and above -55 dBFS).
  SPOT      3 si20k rows: waveform before / after, level, target, gain, peak.
-> <out>/fit.json, hist.png, spot.png. usage: python -I fit_user_levels.py --stores DIR --out DIR [--n-real 3000] [--n-si 1000] [--workers 14]"""
import argparse
import glob
import json
import os
import sys

import numpy as np

CODE = "/data/home/amartyavarma/ppm/si20k-vox-ppm/code"
S2H = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores"
IDX = "/data/home/amartyavarma/ppm/aloe-m1vox-ft/plan/M1VOX_INDEX.json"
FLAC = "/data/home/amartyavarma/ppm/aloe-v2vox-raw/flac"
_S = {}


def _setup():
    if "A" not in _S:
        sys.path.insert(0, CODE); sys.path.insert(0, f"{CODE}/lib")
        os.environ.setdefault("PPM_CD", "2")
        import data_si20k_vox as A
        import audio_s4q as AU
        _S.update(A=A, AU=AU, rows={})
    return _S


def real_one(job):
    k, r, e = job
    S = _setup()
    A, AU = S["A"], S["AU"]
    try:
        if k not in S["rows"]:
            node = A.D.R3.load_node(f"{S2H}/m1n{k}")
            S["rows"][k] = A.D.R3.Rows(node)
        row = S["rows"][k][r]
        assert row["row_key"] == e["row_key"]
        a = e["A"]["user"]
        w = AU.ffmpeg_row_audio({"flac": f"{FLAC}/{a['flac']}", "frame0": int(a["audio_frame0"]), "n_frames": int(a["n_frames"]), "row_key": row["row_key"]}, 0.0)
        lv = A.active_level_db(w, [x["start"] for x in row["words_user"]])
        return {"row_key": row["row_key"], "level": lv, "peak_db": float(20 * np.log10(np.max(np.abs(w)) + 1e-12)), "n_words": len(row["words_user"])}
    except Exception as ex:  # noqa: BLE001
        return {"row_key": e.get("row_key"), "error": repr(ex)[:200]}


def demo_level(path):
    z = np.load(path)
    w = np.asarray(z["a"], np.float64)
    fr = w[: len(w) // 1920 * 1920].reshape(-1, 1920)
    if not len(fr):
        return None
    db = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    keep = (db > np.percentile(db, 95) - 35) & (db > -55)
    if keep.sum() < 10:
        return None
    return float(10 * np.log10((fr[keep] ** 2).mean() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-real", type=int, default=3000); ap.add_argument("--n-si", type=int, default=1000); ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ.update(ALOE_SI20K_STORE_ROOT=a.stores, ALOE_SI20K_SRC=a.stores, ALOE_SI20K_STORES_JSON=f"{a.stores}/STORES.json")
    from multiprocessing import Pool
    idx = json.load(open(IDX))["rows"]
    cand = [(int(k), int(e["row"]), e) for k, rows in idx.items() for e in rows
            if e["row_key"].endswith("/c00") and (e.get("A") or {}).get("user", {}).get("kind") == "gcs"]
    rng = np.random.default_rng(0)
    pick = [cand[i] for i in rng.choice(len(cand), size=min(a.n_real, len(cand)), replace=False)]
    with Pool(a.workers) as pool:
        real = pool.map(real_one, pick, chunksize=4)
    rl = np.array([x["level"] for x in real if x.get("level") is not None])
    err = [x for x in real if "error" in x]
    mean, sd = float(rl.mean()), float(rl.std())
    q = lambda v: {str(p): round(float(np.percentile(v, p)), 2) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}  # noqa: E731
    # demo cross-check
    dl = [demo_level(p) for p in sorted(glob.glob("/data/home/amartyavarma/ppm/aloe-*live*/sessions/*/session.npz"))]
    dl = np.array([x for x in dl if x is not None])
    # si20k rows + the normal draw
    S = _setup(); A = S["A"]
    json.dump({"real": real, "mean": mean, "sd": sd}, open(f"{a.out}/real_levels.json", "w"))
    cfg = {**A.aloe_cfg(), **A.a_gain_cfg("normal"), "a_level_mean": mean, "a_level_sd": sd}
    st = A.Stores()
    items = [(k, r) for k in range(4) for r in range(len(st.open(k)[0]["length"]))]
    sel = [items[i] for i in rng.choice(len(items), size=min(a.n_si, len(items)), replace=False)]
    si, spots = [], []
    for j, (k, r) in enumerate(sel):
        node, rows_, ps, vc, led = st.open(k)
        row = rows_[r]
        w = A.a_audio(led[row["row_key"]])
        on = [x["start"] for x in row["words_user"]]
        lv = A.active_level_db(w, on)
        g, gi = A.gain_db(w, "normal", row["row_key"], on, cfg)
        w2 = (w * np.float32(10 ** (g / 20))).astype(np.float32)
        lv2 = A.active_level_db(w2, on)
        g6, gi6 = A.gain_db(w, "spread:6", row["row_key"], on, cfg)
        lv6 = None if lv is None else lv + g6
        si.append({"row_key": row["row_key"], "level_raw": lv, "target": gi["target_db"], "gain": g, "level_after": lv2, "limited": gi["limited"],
                   "spread6_gain": g6, "spread6_level": lv6, "spread6_limited": gi6["limited"],
                   "no_level": gi["no_level"], "peak_after_db": float(20 * np.log10(np.max(np.abs(w2)) + 1e-12))})
        if j < 3:
            spots.append((row["row_key"], w, w2, si[-1]))
    sl = np.array([x["level_raw"] for x in si if x["level_raw"] is not None])
    sa = np.array([x["level_after"] for x in si if x["level_after"] is not None])
    lim = float(np.mean([x["limited"] for x in si]))
    out = {"real": {"source": "Aloe handover s2h-m1 channel A (data user, SPEAKER_01), first chunks, WLT FLAC 0 dB", "n": int(len(rl)), "errors": len(err),
                    "mean": round(mean, 3), "sd": round(sd, 3), "q": q(rl), "err_head": err[:3]},
           "demo_mic": {"n_sessions": int(len(dl)), "mean": round(float(dl.mean()), 2) if len(dl) else None, "sd": round(float(dl.std()), 2) if len(dl) else None,
                        "q": q(dl) if len(dl) else None, "measure": "energy VAD (no word timings), browser AEC on, AGC off"},
           "si20k_raw": {"n": int(len(sl)), "mean": round(float(sl.mean()), 2), "sd": round(float(sl.std()), 2), "q": q(sl)},
           "draw": {"n": len(si), "after_mean": round(float(sa.mean()), 2), "after_sd": round(float(sa.std()), 2), "after_q": q(sa),
                    "gain_mean": round(float(np.mean([x["gain"] for x in si])), 2), "gain_q": q(np.array([x["gain"] for x in si])),
                    "peak_limited_rate": round(lim, 4), "no_level_rate": round(float(np.mean([x["no_level"] for x in si])), 4),
                    "clip_sd": 2.5, "peak_dbfs": -1.0},
           "spread6": {"after_q": q(np.array([x["spread6_level"] for x in si if x["spread6_level"] is not None])),
                       "peak_limited_rate": round(float(np.mean([x["spread6_limited"] for x in si])), 4)},
           "spots": [s[3] for s in spots]}
    json.dump(out, open(f"{a.out}/fit.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.arange(-75, 0, 1.5)
    ax.hist(rl, bins=bins, alpha=0.5, density=True, label=f"real user mics (Aloe s2h-m1 A, n={len(rl)}): {mean:.1f} ± {sd:.1f} dBFS")
    ax.hist(sa, bins=bins, alpha=0.5, density=True, label=f"si20k after the normal draw (n={len(sa)}): {sa.mean():.1f} ± {sa.std():.1f}")
    ax.hist(sl, bins=bins, histtype="step", lw=1.5, density=True, label=f"si20k raw TTS (n={len(sl)}): {sl.mean():.1f} ± {sl.std():.1f}")
    s6 = np.array([x["spread6_level"] for x in si if x["spread6_level"] is not None])
    ax.hist(s6, bins=bins, histtype="step", lw=1.2, ls=":", density=True, label=f"si20k after spread ±6 dB: {s6.mean():.1f} ± {s6.std():.1f}")
    if len(dl):
        ax.hist(dl, bins=bins, histtype="step", lw=1.5, ls="--", density=True, label=f"live demo mic (n={len(dl)}, energy VAD): {dl.mean():.1f} ± {dl.std():.1f}")
    ax.set_xlabel("user active-speech level (dBFS)"); ax.set_ylabel("density"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{a.out}/hist.png", dpi=110)
    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    for ax, (rk, w, w2, inf) in zip(axs, spots):
        t = np.arange(len(w)) / 24000
        ax.plot(t, w2, lw=0.3, label=f"after (gain {inf['gain']:+.1f} dB -> {inf['level_after']:.1f} dBFS, target {inf['target']:.1f})")
        ax.plot(t, w, lw=0.3, label=f"raw ({inf['level_raw']:.1f} dBFS)")
        ax.set_title(rk, fontsize=8); ax.legend(fontsize=7, loc="upper right"); ax.set_ylim(-1, 1)
    fig.tight_layout(); fig.savefig(f"{a.out}/spot.png", dpi=100)
    print(json.dumps({k: v for k, v in out.items() if k != "spots"}, indent=1))
    print("spots", json.dumps(out["spots"]))


if __name__ == "__main__":
    main()
