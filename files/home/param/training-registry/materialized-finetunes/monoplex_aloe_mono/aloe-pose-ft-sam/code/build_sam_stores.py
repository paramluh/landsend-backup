"""aloe-pose-ft-sam STORE BUILD (10-26): the Aloe handover pose stores of job 22091 (ppm/aloe-pose-ft/stores p0..p7, read only) with ONLY
the Mimi codes arrays swapped for the SAM-separated, crossbleed-removed per-speaker ISOLATED stems of HF InternalCan/stage2-sam-sep-v1-mimi
@ecb72b2d (local copy data/aloe-sam-sep/hf; same mimi-stream-v5 recipe, same (session, seg) keys, same T; README there):
  codes_assistant <- SPEAKER_00_isolated_npy     codes_user <- SPEAKER_01_isolated_npy
  (roles = build_poseB_rows.py: SPEAKER_00 = data 'assistant', SPEAKER_01 = data 'user'; asserted per row from rows.parquet)
  each row = the row's own window [frame_start, frame_start + n_frames) of the conversation, PAD_CODE (-1) after n_frames (as built).
Every other store file is copied byte for byte (sha256 asserted). Asserted per row: (session, seg) key in the SAM index with equal_T, SAM T ==
the row's conversation T (src_frames), SAM stream int16 [32, T] in 0..2047, window inside T, original pad region all -1, original row ==
original layout. A row whose SAM codes are missing -> `check` exits 1 and nothing is built (never dropped, never a fallback).
Voice clips need no change: the trainer cuts them from B's own codes (data_ppm.voice_clips), so they are SAM codes too.
usage: python -I build_sam_stores.py cache --shard I        (SAM parquet shard I -> _samcache/<session>__<seg>.npy int16 [2, 32, T])
       python -I build_sam_stores.py check                  (every store row has its SAM pair, equal T; CHECK.json; exit 1 on any miss)
       python -I build_sam_stores.py build --store K --workers N
       python -I build_sam_stores.py report                 (per-codebook differing-frame fractions -> SAM_BUILD.json)"""
import argparse
import collections
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
import pyarrow.parquet as pq

SRC = "/data/home/amartyavarma/ppm/aloe-pose-ft/stores"
OUT = "/data/home/amartyavarma/ppm/aloe-pose-ft-sam"
DST = f"{OUT}/stores"
CACHE = f"{OUT}/_samcache"
SAM = "/data/home/amartyavarma/data/aloe-sam-sep/hf"
SAM_REV = "InternalCan/stage2-sam-sep-v1-mimi@ecb72b2de2347bfbbc3b71b0236e2986b0c44572"
LANE = {"assistant": ("SPEAKER_00", "SPEAKER_00_isolated_npy", 0), "user": ("SPEAKER_01", "SPEAKER_01_isolated_npy", 1)}
SIDES = ("user", "assistant")                                     # = the stores' channel order (NODE.json "channels")
SHARD = "rows-00000-of-00001"
DEPTH, RF, PAD, CBS = 32, 7500, -1, 2048
NSTORE = 8


os.environ["TZ"] = "America/Los_Angeles"; time.tzset()


def pt():
    return time.strftime("%m-%d %H:%M:%S", time.localtime()) + " PT"


def log(*a):
    print(f"[{pt()}]", *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cache_path(conv):
    ses, seg = conv.split("/")
    return f"{CACHE}/{ses}__{seg}.npy"


def store_rows(k):
    t = pq.read_table(f"{SRC}/p{k}/rows.parquet").to_pylist()
    for i, r in enumerate(t):
        assert r["row"] == i and r["row_in_shard"] == i and r["shard"] == SHARD, (k, i, r["row"], r["row_in_shard"], r["shard"])
        assert r["assistant_speaker_dir"] == "SPEAKER_00" and r["user_speaker_dir"] == "SPEAKER_01", (k, i, r["row_key"])
        ses, seg = r["src_conversation"].split("/")
        assert r["row_key"] == f"ALOE/s2handover-pose/{ses}/{seg}/c{r['chunk_index']:02d}", (k, i, r["row_key"])
        assert r["frame_end"] - r["frame_start"] == r["n_frames"] and r["frame_start"] == RF * r["chunk_index"], (k, i, r["row_key"])
    return t


def needed():
    return sorted({r["src_conversation"] for k in range(NSTORE) for r in store_rows(k)})


def cmd_cache(a):
    shards = sorted(glob.glob(f"{SAM}/tokens/*.parquet"))
    sh = shards[a.shard]
    need = set(needed())
    os.makedirs(CACHE, exist_ok=True)
    t0, got = time.time(), {}
    f = pq.ParquetFile(sh)
    for g in range(f.metadata.num_row_groups):
        t = f.read_row_group(g, columns=["session", "seg", "T", "equal_T", "SPEAKER_00_isolated_npy", "SPEAKER_01_isolated_npy"])
        for r in t.to_pylist():
            conv = f"{r['session']}/{r['seg']}"
            if conv not in need:
                continue
            T = int(r["T"])
            arr = []
            for col in ("SPEAKER_00_isolated_npy", "SPEAKER_01_isolated_npy"):
                x = np.load(io.BytesIO(r[col]), allow_pickle=False)
                assert x.dtype == np.int16 and x.shape == (DEPTH, T), (conv, col, x.dtype, x.shape, T)
                assert int(x.min()) >= 0 and int(x.max()) < CBS, (conv, col, int(x.min()), int(x.max()))
                arr.append(x)
            assert r["equal_T"], (conv, "equal_T False")
            out = cache_path(conv)
            np.save(out + ".tmp.npy", np.stack(arr))
            os.replace(out + ".tmp.npy", out)
            got[conv] = T
    json.dump({"shard": os.path.basename(sh), "convs": got}, open(f"{CACHE}/shard{a.shard:02d}.done.json.tmp", "w"))
    os.replace(f"{CACHE}/shard{a.shard:02d}.done.json.tmp", f"{CACHE}/shard{a.shard:02d}.done.json")
    log(f"[cache] shard {a.shard} {os.path.basename(sh)}: {len(got)} needed conversations cached ({time.time() - t0:.0f}s)")


def cmd_check(a):
    pairs = {f"{r['session']}/{r['seg']}": r for r in pq.read_table(f"{SAM}/index/pairs.parquet",
             columns=["session", "seg", "T", "equal_T", "token_shard", "sam_sep_flags", "mimi_flags", "n_swapped_chunks",
                      "n_ambiguous_chunks"]).to_pylist()}
    cached = {}
    for f in sorted(glob.glob(f"{CACHE}/shard*.done.json")):
        cached.update(json.load(open(f))["convs"])
    miss, bad, rows, convs = [], [], 0, {}
    for k in range(NSTORE):
        for r in store_rows(k):
            rows += 1
            conv = r["src_conversation"]; convs[conv] = r["src_frames"]
            p = pairs.get(conv)
            if p is None:
                miss.append({"store": k, "row": r["row"], "row_key": r["row_key"], "why": "no SAM pair"}); continue
            if not p["equal_T"] or int(p["T"]) != int(r["src_frames"]) or r["frame_end"] > int(p["T"]):
                bad.append({"store": k, "row": r["row"], "row_key": r["row_key"], "sam_T": int(p["T"]), "src_frames": r["src_frames"],
                            "frame_end": r["frame_end"], "equal_T": p["equal_T"]}); continue
            if cached.get(conv) != int(p["T"]) or not os.path.exists(cache_path(conv)):
                miss.append({"store": k, "row": r["row"], "row_key": r["row_key"], "why": "SAM pair not cached"})
    fl = collections.Counter()
    for conv in convs:
        p = pairs.get(conv)
        if p:
            fl["sam_sep_flags"] += p["sam_sep_flags"] not in ("", "[]", None); fl["mimi_flags"] += p["mimi_flags"] not in ("", "[]", None)
            fl["swapped_chunk"] += int(p["n_swapped_chunks"] or 0) > 0; fl["ambiguous_chunk"] += int(p["n_ambiguous_chunks"] or 0) > 0
    res = {"rows": rows, "conversations": len(convs), "missing": miss, "T_mismatch": bad, "qc_flags_in_used_convs": dict(fl),
           "time": pt()}
    json.dump(res, open(f"{OUT}/CHECK.json", "w"), indent=1)
    log(f"[check] {rows} rows, {len(convs)} conversations: missing {len(miss)}, T mismatch {len(bad)}; QC flags {dict(fl)}")
    sys.exit(1 if (miss or bad) else 0)


_W = {}


def _winit(k, tmp):
    _W["rows"] = store_rows(k)
    _W["orig"] = {s: np.load(f"{SRC}/p{k}/{SHARD}.codes_{s}.npy", mmap_mode="r") for s in SIDES}
    _W["new"] = {s: np.load(f"{tmp}/{SHARD}.codes_{s}.npy", mmap_mode="r+") for s in SIDES}


def _wrows(rng):
    """rows [lo, hi): write the SAM rows, then read them back. -> (lo, diff [n, 2, 32] int64, nfr [n], cb0 [n, 2, 2] (own, cross) matches)"""
    lo, hi = rng
    rows, orig, new = _W["rows"], _W["orig"], _W["new"]
    diff = np.zeros((hi - lo, 2, DEPTH), np.int64); nfr = np.zeros(hi - lo, np.int64); cb0 = np.zeros((hi - lo, 2, 2), np.int64)
    sam = None; conv0 = None
    for i in range(lo, hi):
        r = rows[i]; n, fs = int(r["n_frames"]), int(r["frame_start"])
        if r["src_conversation"] != conv0:
            conv0 = r["src_conversation"]; sam = np.load(cache_path(conv0), mmap_mode="r")
            assert sam.shape == (2, DEPTH, int(r["src_frames"])), (r["row_key"], sam.shape, r["src_frames"])
        o = {s: np.asarray(orig[s][i]) for s in SIDES}
        for j, s in enumerate(SIDES):
            lane = LANE[s][2]
            w = np.asarray(sam[lane, :, fs:fs + n])
            assert w.shape == (DEPTH, n), (r["row_key"], s, w.shape, n)
            assert (o[s][:, n:] == PAD).all() and int(o[s][:, :n].min()) >= 0 and int(o[s][:, :n].max()) < CBS, (r["row_key"], s, "orig layout")
            row = np.full((DEPTH, RF), PAD, np.int16); row[:, :n] = w
            new[s][i] = row
            diff[i - lo, j] = (o[s][:, :n] != w).sum(1)
            other = SIDES[1 - j]
            cb0[i - lo, j] = ((o[s][0, :n] == w[0]).sum(), (o[other][0, :n] == w[0]).sum())
        nfr[i - lo] = n
    for s in SIDES:
        new[s].flush()
    for i in range(lo, hi):                                          # read back what was written
        r = rows[i]; n, fs = int(r["n_frames"]), int(r["frame_start"])
        sam = np.load(cache_path(r["src_conversation"]), mmap_mode="r")
        for s in SIDES:
            x = new[s][i]
            assert (x[:, n:] == PAD).all() and np.array_equal(x[:, :n], sam[LANE[s][2], :, fs:fs + n]), (r["row_key"], s, "readback")
    return lo, diff, nfr, cb0


def _sha_pair(job):
    a, b = job
    return os.path.basename(a), sha256(a), (sha256(b) if b else None)


def cmd_build(a):
    k = a.store
    src, out, tmp = f"{SRC}/p{k}", f"{DST}/p{k}", f"{DST}/p{k}.tmp"
    assert not os.path.exists(out), (out, "exists")
    t0 = time.time()
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    rows = store_rows(k)
    R = len(rows)
    for s in SIDES:
        o = np.load(f"{src}/{SHARD}.codes_{s}.npy", mmap_mode="r")
        assert o.shape == (R, DEPTH, RF) and o.dtype == np.int16, (k, s, o.shape, o.dtype)
        m = np.lib.format.open_memmap(f"{tmp}/{SHARD}.codes_{s}.npy", mode="w+", dtype=np.int16, shape=(R, DEPTH, RF))
        del m
    other = sorted(f for f in os.listdir(src) if not f.startswith(f"{SHARD}.codes_"))
    for f in other:
        shutil.copyfile(f"{src}/{f}", f"{tmp}/{f}")
    log(f"[p{k}] {R} rows; {len(other)} other files copied ({time.time() - t0:.0f}s)")
    step = max(1, -(-R // (a.workers * 4)))
    rngs = [(lo, min(lo + step, R)) for lo in range(0, R, step)]
    diff = np.zeros((R, 2, DEPTH), np.int64); nfr = np.zeros(R, np.int64); cb0 = np.zeros((R, 2, 2), np.int64)
    with mp.get_context("fork").Pool(a.workers, initializer=_winit, initargs=(k, tmp)) as pool:
        for j, (lo, d, n, c) in enumerate(pool.imap_unordered(_wrows, rngs)):
            diff[lo:lo + len(n)] = d; nfr[lo:lo + len(n)] = n; cb0[lo:lo + len(n)] = c
        log(f"[p{k}] codes written + read back ({time.time() - t0:.0f}s)")
        jobs = [(f"{src}/{f}", f"{tmp}/{f}") for f in other] + [(f"{tmp}/{SHARD}.codes_{s}.npy", None) for s in SIDES]
        shas = {name: (h1, h2) for name, h1, h2 in pool.imap_unordered(_sha_pair, jobs)}
    for f in other:
        assert shas[f][0] == shas[f][1], (k, f, "copy sha256 differs")
    assert (nfr == np.array([r["n_frames"] for r in rows])).all()
    np.savez(f"{tmp}/sam_rowstats.npz", diff=diff, nfr=nfr, cb0=cb0)
    json.dump({"format": "aloe-pose-ft-sam-v1", "store": f"p{k}", "rows": R, "source": SAM_REV, "streams": {s: LANE[s][1] for s in SIDES},
               "from": src, "copied_byte_identical_sha256": {f: shas[f][0] for f in other},
               "codes_sha256": {s: shas[f"{SHARD}.codes_{s}.npy"][0] for s in SIDES},
               "builder": os.path.abspath(__file__), "builder_sha256": sha256(os.path.abspath(__file__)),
               "built": pt()}, open(f"{tmp}/SAM.json", "w"), indent=1)
    os.replace(tmp, out)
    open(f"{out}/SAM_BUILD_DONE", "w").write(str(R))
    log(f"[p{k}] DONE -> {out}: {R} rows ({time.time() - t0:.0f}s)")


def cmd_report(a):
    agg = collections.defaultdict(lambda: [np.zeros(DEPTH, np.int64), 0])
    viol = collections.Counter(); conv_cb0 = collections.defaultdict(lambda: np.zeros(2, np.int64))
    for k in range(NSTORE):
        z = np.load(f"{DST}/p{k}/sam_rowstats.npz"); rows = store_rows(k)
        pi = np.load(f"{SRC}/p{k}/pose_index.npz")
        for i, r in enumerate(rows):
            sp = "eval" if int(pi["split"][i]) else "train"
            for j, s in enumerate(SIDES):
                n = int(z["nfr"][i])
                for g in ("all rows, both lanes", f"B tracks {sp}" if pi["b_ok"][i, j] else None):
                    if g:
                        agg[g][0] += z["diff"][i, j]; agg[g][1] += n
                conv_cb0[(r["src_conversation"], s)] += z["cb0"][i, j]                 # (own lane, other lane) cb0 matches
                viol["rows (row, side) own cb0 match <= cross"] += int(z["cb0"][i, j, 0] <= z["cb0"][i, j, 1])
    for (conv, s), m in conv_cb0.items():
        viol["conversation sides own cb0 match <= cross"] += int(m[0] <= m[1])
    viol["conversation sides"] = len(conv_cb0)
    rep = {"source": SAM_REV, "per_codebook_differing_frame_fraction": {
        g: {"frames": int(n), "all_cb": round(float(d.sum() / (n * DEPTH)), 4), "per_cb": [round(float(x / n), 4) for x in d]}
        for g, (d, n) in sorted(agg.items())}, "speaker_sanity": dict(viol), "check": json.load(open(f"{OUT}/CHECK.json")), "time": pt()}
    json.dump(rep, open(f"{OUT}/SAM_BUILD.json", "w"), indent=1)
    for g, v in rep["per_codebook_differing_frame_fraction"].items():
        log(f"[report] {g}: frames {v['frames']:,} all-cb {v['all_cb']:.4f} cb0..3 {v['per_cb'][:4]} cb4..31 {min(v['per_cb'][4:])}-{max(v['per_cb'][4:])}")
    log(f"[report] speaker sanity {dict(viol)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("cache", "check", "build", "report"))
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--store", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"cache": cmd_cache, "check": cmd_check, "build": cmd_build, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
