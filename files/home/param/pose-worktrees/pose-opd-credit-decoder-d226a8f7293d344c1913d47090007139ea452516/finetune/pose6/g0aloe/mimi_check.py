"""pose6 / B2 (10-27): Mimi recipe check, metadata census + cb0 histogram spot check (read-only, CPU).
(a) 31101 pose rows: fresh plan 1 (new-pose-llm-data, /data copy) + npl5 (delivery 5, npl5-weka window); npl2/npl3 are node-local only.
(b) Aloe pose stores: per-conversation w00.mimi.json (mimi-stream-v5) of every B track in g0aloe.parquet + the store codes.
(c) engine Mimi weights: sha256 of /data/home/amartyavarma/models/mimi/model.safetensors.
Writes pose6/g0aloe/mimi_check.json. usage: python -I mimi_check.py"""
import collections
import glob
import hashlib
import json
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

OUT = "/data/home/amartyavarma/ppm/pose6/g0aloe"
FRESH_ROOT = "/data/home/amartyavarma/data/new-pose-llm-data"
FRESH_TSV = "/data/home/amartyavarma/data/new-pose-llm-data-dl/PLACEMENT_files.tsv"
NPL5 = "/data/home/amartyavarma/ppm/pose5/npl5-weka"
MIMICHK = "/data/home/amartyavarma/ppm/pose5/code-811-s2p/fresh/mimichk"
STORES = "/data/home/amartyavarma/ppm/aloe-pose-ft/stores"
MIMI_W = "/data/home/amartyavarma/models/mimi/model.safetensors"
HF_SHA = "bac7e85083dcded655d24eaadde7e6eea34c0da1b35fa2d284e641bd2b942a5e"   # kyutai/mimi@89091b3e model.safetensors (HF blob name)
rng = random.Random(20261027)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def hist(c):
    return np.bincount(np.asarray(c, np.int64).ravel(), minlength=2048)[:2048].astype(np.float64)


def jsd(a, b):
    p, q = a / a.sum(), b / b.sum()
    m = (p + q) / 2
    kl = lambda x, y: float(np.sum(np.where(x > 0, x * np.log2(np.where(x > 0, x, 1) / np.where(y > 0, y, 1)), 0)))  # noqa: E731
    return round(0.5 * kl(p, m) + 0.5 * kl(q, m), 5)


def tv(a, b):
    return round(0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum()), 5)


def pct(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], np.float64)
    return {"n": int(len(x)), **({f"p{q}": round(float(np.percentile(x, q)), 3) for q in (1, 5, 25, 50, 75, 95, 99)} if len(x) else {})}


def parquet_sample(paths, max_rows_per_file=200):
    """-> (segmeta audio_processing census, cb0 per segment list, achieved LUFS list, lengths)."""
    import pyarrow.parquet as pq
    cen, cb0, ach, short, lens, nfile = collections.Counter(), [], [], [], [], 0
    for p in paths:
        try:
            t = pq.read_table(p, columns=["segment_metadata_json", "audio"])
        except Exception as e:  # noqa: BLE001
            cen[f"read_error {type(e).__name__}"] += 1
            continue
        nfile += 1
        sm = t.column("segment_metadata_json").to_pylist()[:max_rows_per_file]
        au = t.column("audio").to_pylist()[:max_rows_per_file]
        for s, a in zip(sm, au):
            s = json.loads(s) if isinstance(s, str) else (s or {})
            ap = s.get("audio_processing") or {}
            ld = ap.get("loudness") or {}
            cen[("recipe", ap.get("recipe"))] += 1
            cen[("attention_policy", ap.get("attention_policy"))] += 1
            cen[("normalization_scope", ap.get("normalization_scope"))] += 1
            cen[("lufs_target", ld.get("lufs_target"))] += 1
            cen[("peak_ceiling", ap.get("peak_ceiling"))] += 1
            cen[("lufs_status", ld.get("lufs_status"))] += 1
            cen[("audio_encode", s.get("audio_encode"))] += 1
            cen[("audio_sr", a.get("audio_sr"))] += 1
            if ld.get("lufs_in") is not None and ld.get("lufs_gain_db") is not None:
                ach.append(float(ld["lufs_in"]) + float(ld["lufs_gain_db"]))
                if ld.get("lufs_gain_db_intended") is not None:
                    short.append(float(ld["lufs_gain_db_intended"]) - float(ld["lufs_gain_db"]))
            mc = a.get("mimi_codes")
            if mc:
                cb0.append(np.asarray(mc[0], np.int64)); lens.append(len(mc[0]))
    return {"files_read": nfile, "census": {f"{k[0]}={k[1]}" if isinstance(k, tuple) else k: v for k, v in sorted(cen.items(), key=str)},
            "achieved_lufs": pct(ach), "gain_shortfall_db": pct(short), "seg_frames": pct(lens)}, cb0


def main():
    res = {}
    # ------------------------------------------------------------------------------------------- (c) weights
    h = hashlib.sha256()
    with open(MIMI_W, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    res["engine_mimi_dir_safetensors_sha256"] = h.hexdigest()
    res["engine_mimi_dir_equals_kyutai_89091b3e"] = h.hexdigest() == HF_SHA
    log("weights", res["engine_mimi_dir_equals_kyutai_89091b3e"])
    # ------------------------------------------------------------------------------------------- (a) plan 1: runtime.json + parquet
    lines = open(FRESH_TSV).read().split("\n")
    pq_lines = [l.split("\t")[0] for l in lines if l.split("\t")[0].endswith(".parquet")]
    rng.shuffle(pq_lines)
    rels = []
    for rel in pq_lines:
        d = os.path.dirname(f"{FRESH_ROOT}/{rel}")
        if os.path.exists(f"{d}/runtime.json"):
            rels.append(d)
        if len(rels) >= 400:
            break
    rt = collections.Counter()

    def rd_rt(d):
        j = json.load(open(f"{d}/runtime.json"))
        c, m, pk = j.get("config") or {}, (j.get("models") or {}).get("mimi_model_id") or {}, j.get("packages") or {}
        return (c.get("mimi_audio_recipe"), c.get("mimi_max_encode_s"), c.get("dtype"), m.get("model.safetensors"), m.get("config.json"),
                pk.get("transformers"), pk.get("torch"), pk.get("pyloudnorm"))
    with ThreadPoolExecutor(16) as ex:
        for k in ex.map(rd_rt, rels):
            rt[k] += 1
    res["plan1_runtime"] = {"releases_sampled": len(rels), "keys": "(mimi_audio_recipe, mimi_max_encode_s, dtype, mimi safetensors sha256, "
                            "mimi config sha256, transformers, torch, pyloudnorm)", "census": [[list(k), v] for k, v in rt.most_common()]}
    log("plan1 runtime", rt.most_common(3))
    p1_files = [f"{d}/{x}" for d in rels[:60] for x in sorted(os.listdir(d)) if x.endswith(".parquet")][:60]
    res["plan1_parquet"], cb0_p1 = parquet_sample(p1_files)
    log("plan1 parquet", res["plan1_parquet"]["files_read"], len(cb0_p1))
    # ------------------------------------------------------------------------------------------- (a) npl5 (delivery 5)
    n5 = sorted(glob.glob(f"{NPL5}/*/jobs/*/*/part-*.parquet"))
    rng.shuffle(n5)
    res["npl5_parquet"], cb0_n5 = parquet_sample(n5[:60])
    res["npl5_parquet"]["files_available"] = len(n5)
    n5rt = sorted(glob.glob(f"{NPL5}/*/jobs/*/*/runtime.json"))
    res["npl5_runtime_json_on_disk"] = len(n5rt)
    log("npl5 parquet", res["npl5_parquet"]["files_read"], len(cb0_n5))
    # ------------------------------------------------------------------------------------------- (b) Aloe: mimi.json of every B track
    import pyarrow.parquet as pq
    g = pq.read_table(f"{OUT}/g0aloe.parquet", columns=["pose_json", "speaker", "store", "row", "b_side", "n_frames", "conversation"]).to_pylist()
    tracks = sorted({(os.path.dirname(r["pose_json"]), r["speaker"]) for r in g})
    enc_c, st_c, ach, short = collections.Counter(), collections.Counter(), [], []

    def rd_mj(t):
        d, spk = t
        j = json.load(open(f"{d}/w00.mimi.json"))
        e, m = j.get("encoder") or {}, (j.get("mimi") or {}).get(spk) or {}
        return ((e.get("model"), e.get("model_revision"), e.get("encoder_version"), e.get("dtype"), e.get("lufs_target"), e.get("lufs_method"),
                 e.get("sliding_window_encoder_frames"), e.get("sliding_window_seconds"), e.get("api"), e.get("chunk_schedule"),
                 e.get("transformers_version"), e.get("max_encode_s")),
                m.get("lufs_status"), m.get("lufs_in"), m.get("lufs_gain_db"), m.get("lufs_gain_db_intended"), m.get("audio_sr"))
    with ThreadPoolExecutor(16) as ex:
        for e, s, li, gd, gi, sr in ex.map(rd_mj, tracks):
            enc_c[e] += 1; st_c[(s, sr)] += 1
            if li is not None and gd is not None:
                ach.append(float(li) + float(gd))
                if gi is not None:
                    short.append(float(gi) - float(gd))
    res["aloe_mimi_json"] = {"b_tracks": len(tracks), "encoder_keys": "(model, model_revision, encoder_version, dtype, lufs_target, lufs_method, "
                             "sliding_window_frames, sliding_window_s, api, chunk_schedule, transformers, max_encode_s)",
                             "encoder_census": [[list(k), v] for k, v in enc_c.most_common()],
                             "status_sr_census": [[list(k), v] for k, v in st_c.most_common()],
                             "achieved_lufs_track": pct(ach), "gain_shortfall_db": pct(short),
                             "frac_peak_limited_gt_0p5db": round(float(np.mean(np.asarray(short) > 0.5)), 4) if short else None}
    log("aloe mimi.json", len(tracks), enc_c.most_common(1))
    # Aloe B codes from the stores (what the FT feeds): 400 random items, all their frames
    sel = rng.sample(g, 400)
    cb0_al, conv = [], []
    mm = {}
    for r in sel:
        key = (r["store"], r["b_side"])
        if key not in mm:
            mm[key] = np.load(f"{STORES}/p{r['store']}/rows-00000-of-00001.codes_{r['b_side']}.npy", mmap_mode="r")
        c = np.asarray(mm[key][r["row"], 0, :r["n_frames"]]).astype(np.int64)
        assert (c >= 0).all()
        cb0_al.append(c); conv.append(r["conversation"])
    # ------------------------------------------------------------------------------------------- recipe-only calibration: same clips, old vs fresh
    fresh_same = [x for f in sorted(glob.glob(f"{MIMICHK}/fresh-*.pkl")) for x in pickle.load(open(f, "rb"))]
    tk = pickle.load(open(f"{MIMICHK}/tiktok.pkl", "rb"))
    byc = collections.defaultdict(list)
    for r in tk:
        byc[r["clip_id"]].append(r)
    hs_f, hs_t = np.zeros(2048), np.zeros(2048)
    hs_f10, hs_t10 = np.zeros(2048), np.zeros(2048)
    npair = 0
    for x in fresh_same:
        k = int(x["segment_index"])
        cand = [r for r in byc.get(x["clip_id"], []) if int(r.get("segment", -1) if r.get("segment") is not None else -1) in ((k, -1) if k == 0 else (k,))]
        if not cand:
            continue
        a, b = np.asarray(x["mimi"])[0], np.asarray(cand[0]["mimi"])[0]
        n = min(len(a), len(b))
        hs_f += hist(a[:n]); hs_t += hist(b[:n]); npair += 1
        if n > 125:
            hs_f10 += hist(a[125:n]); hs_t10 += hist(b[125:n])
    # ------------------------------------------------------------------------------------------- histograms
    def cat(xs):
        return np.concatenate(xs) if xs else np.zeros(0, np.int64)
    H = {"plan1": hist(cat(cb0_p1)), "npl5": hist(cat(cb0_n5)), "aloeB": hist(cat(cb0_al))}
    half = lambda xs: (hist(cat(xs[0::2])), hist(cat(xs[1::2])))  # noqa: E731
    p1a, p1b = half(cb0_p1)
    aorder = sorted(range(len(cb0_al)), key=lambda i: conv[i])
    ala, alb = hist(cat([cb0_al[i] for i in aorder[:200]])), hist(cat([cb0_al[i] for i in aorder[200:]]))
    res["cb0_hist"] = {
        "frames": {k: int(v.sum()) for k, v in H.items()} | {"same_clip_pairs": npair, "same_clip_frames": int(hs_f.sum())},
        "JSD_bits(plan1, npl5)": jsd(H["plan1"], H["npl5"]), "TV(plan1, npl5)": tv(H["plan1"], H["npl5"]),
        "JSD_bits(plan1, aloeB)": jsd(H["plan1"], H["aloeB"]), "TV(plan1, aloeB)": tv(H["plan1"], H["aloeB"]),
        "JSD_bits(npl5, aloeB)": jsd(H["npl5"], H["aloeB"]), "TV(npl5, aloeB)": tv(H["npl5"], H["aloeB"]),
        "floor JSD_bits(plan1 half A, half B)": jsd(p1a, p1b), "floor JSD_bits(aloeB convs A, convs B)": jsd(ala, alb),
        "calib JSD_bits(same clips: fresh lufs23-local vs old TikTok encode)": jsd(hs_f, hs_t),
        "calib TV(same clips)": tv(hs_f, hs_t),
        "calib JSD_bits(same clips, frames >= 10 s)": jsd(hs_f10, hs_t10) if hs_f10.sum() else None,
        "top5_share": {k: round(float(np.sort(v)[::-1][:5].sum() / v.sum()), 4) for k, v in H.items()},
        "top5_codes": {k: np.argsort(v)[::-1][:5].tolist() for k, v in H.items()},
    }
    log("hist", json.dumps(res["cb0_hist"])[:600])
    res["wall_s"] = round(time.time() - T0, 1); res["host"] = os.uname().nodename
    json.dump(res, open(f"{OUT}/mimi_check.json", "w"), indent=1)
    log("wrote mimi_check.json")


if __name__ == "__main__":
    main()
