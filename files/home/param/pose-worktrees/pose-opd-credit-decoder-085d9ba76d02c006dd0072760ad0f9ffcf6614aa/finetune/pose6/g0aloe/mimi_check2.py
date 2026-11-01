"""pose6 / B2 (10-27): content-controlled cb0 histogram check: only WORD-ACTIVE frames of the speaker (Aloe B tracks spend much of the time
listening -> silence codes dominate an unmasked histogram). Fresh plan 1 + npl5 segments: audio.word_start/end_seconds (segment-local);
Aloe B items: the store's words_<side> start_frame..end_frame. Writes pose6/g0aloe/mimi_check2.json. usage: python -I mimi_check2.py"""
import glob
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/ppm/pose6/g0aloe")
import mimi_check as MC  # noqa: E402  (hist / jsd / tv / paths; its main() is not run)

OUT = MC.OUT
FPS = 12.5
rng = random.Random(20261027)
T0 = time.time()


def fresh_word_frames(paths, max_rows=200):
    import pyarrow.parquet as pq
    act, allf, nf = [], [], 0
    for p in paths:
        try:
            t = pq.read_table(p, columns=["audio"])
        except Exception:  # noqa: BLE001
            continue
        nf += 1
        for a in t.column("audio").to_pylist()[:max_rows]:
            mc = a.get("mimi_codes")
            if not mc:
                continue
            c = np.asarray(mc[0], np.int64); T = len(c)
            m = np.zeros(T, bool)
            for s, e in zip(a.get("word_start_seconds") or [], a.get("word_end_seconds") or []):
                if s is None or e is None:
                    continue
                lo, hi = max(0, int(np.floor(s * FPS))), min(T, int(np.floor(e * FPS)) + 1)
                m[lo:hi] = True
            act.append(c[m]); allf.append(c)
    return act, allf, nf


def main():
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    res = {}
    # fresh plan 1: parquet files that exist on /data (sampled from the placement listing)
    lines = [l.split("\t")[0] for l in open(MC.FRESH_TSV).read().split("\n") if l.split("\t")[0].endswith(".parquet")]
    rng.shuffle(lines)
    p1 = []
    for rel in lines:
        f = f"{MC.FRESH_ROOT}/{rel}"
        if os.path.exists(f):
            p1.append(f)
        if len(p1) >= 120:
            break
    a1, f1, n1 = fresh_word_frames(p1)
    n5 = sorted(glob.glob(f"{MC.NPL5}/*/jobs/*/*/part-*.parquet")); rng.shuffle(n5)
    a5, f5, nn5 = fresh_word_frames(n5[:60])
    print(f"[{time.time() - T0:.1f}s] plan1 files {n1} segs {len(a1)}; npl5 files {nn5} segs {len(a5)}", flush=True)
    # Aloe B items: 400 random items of g0aloe.parquet (seeded; not the same draw as mimi_check)
    g = pq.read_table(f"{OUT}/g0aloe.parquet", columns=["pose_json", "speaker", "store", "row", "b_side", "n_frames", "conversation"]).to_pylist()
    rng2 = random.Random(20261027)
    sel = rng2.sample(g, 400)
    mm, tb = {}, {}
    aa, af = [], []
    for r in sel:
        k, side = r["store"], r["b_side"]
        if (k, side) not in mm:
            mm[(k, side)] = np.load(f"{MC.STORES}/p{k}/rows-00000-of-00001.codes_{side}.npy", mmap_mode="r")
        if k not in tb:
            tb[k] = ipc.open_file(pa.memory_map(f"{MC.STORES}/p{k}/rows-00000-of-00001.words.arrow", "r")).read_all()
        n = r["n_frames"]
        c = np.asarray(mm[(k, side)][r["row"], 0, :n]).astype(np.int64)
        ws = tb[k].slice(r["row"], 1).column(f"words_{side}").to_pylist()[0] or []
        m = np.zeros(n, bool)
        for w in ws:
            m[max(0, int(w["start_frame"])):min(n, int(w["end_frame"]) + 1)] = True
        aa.append(c[m]); af.append(c)
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int64)  # noqa: E731
    H = {"plan1_words": MC.hist(cat(a1)), "npl5_words": MC.hist(cat(a5)), "aloeB_words": MC.hist(cat(aa))}
    res["word_active_frac"] = {"plan1": round(len(cat(a1)) / max(1, len(cat(f1))), 4), "npl5": round(len(cat(a5)) / max(1, len(cat(f5))), 4),
                               "aloeB": round(len(cat(aa)) / max(1, len(cat(af))), 4)}
    half = lambda xs: (MC.hist(cat(xs[0::2])), MC.hist(cat(xs[1::2])))  # noqa: E731
    h5a, h5b = half(a5)
    haa, hab = half(aa)
    res["cb0_hist_word_active"] = {
        "frames": {k: int(v.sum()) for k, v in H.items()},
        "JSD_bits(plan1, npl5)": MC.jsd(H["plan1_words"], H["npl5_words"]),
        "JSD_bits(plan1, aloeB)": MC.jsd(H["plan1_words"], H["aloeB_words"]), "TV(plan1, aloeB)": MC.tv(H["plan1_words"], H["aloeB_words"]),
        "JSD_bits(npl5, aloeB)": MC.jsd(H["npl5_words"], H["aloeB_words"]), "TV(npl5, aloeB)": MC.tv(H["npl5_words"], H["aloeB_words"]),
        "floor JSD_bits(npl5 half A, half B)": MC.jsd(h5a, h5b), "floor JSD_bits(aloeB items A, items B)": MC.jsd(haa, hab),
        "top5_share": {k: round(float(np.sort(v)[::-1][:5].sum() / v.sum()), 4) for k, v in H.items()},
        "top5_codes": {k: np.argsort(v)[::-1][:5].tolist() for k, v in H.items()},
        "aloe_silence_codes_share_in_fresh_words": None,
    }
    # where do Aloe's top-5 (all-frame, mostly listening) codes sit in the fresh word-active histogram?
    hall = MC.hist(cat(af))
    top_al = np.argsort(hall)[::-1][:5]
    res["cb0_hist_word_active"]["aloe_silence_codes_share_in_fresh_words"] = {
        "aloe_top5_all_frames": top_al.tolist(), "share_in_aloe_all": round(float(hall[top_al].sum() / hall.sum()), 4),
        "share_in_aloe_words": round(float(H["aloeB_words"][top_al].sum() / H["aloeB_words"].sum()), 4),
        "share_in_npl5_words": round(float(H["npl5_words"][top_al].sum() / H["npl5_words"].sum()), 4),
        "share_in_npl5_all": round(float(MC.hist(cat(f5))[top_al].sum() / max(1, len(cat(f5)))), 4)}
    res["wall_s"] = round(time.time() - T0, 1); res["host"] = os.uname().nodename
    json.dump(res, open(f"{OUT}/mimi_check2.json", "w"), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
