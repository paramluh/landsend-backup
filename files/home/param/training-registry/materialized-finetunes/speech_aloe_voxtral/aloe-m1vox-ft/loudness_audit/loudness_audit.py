"""LOUDNESS AUDIT (10-17) of channel A's waveform as fed to the frozen Voxtral encoder in aloe-m1vox-ft (job 22335) and aloe-v2vox-ft (job 22420).

What reaches Voxtral (read from the code, nothing else added):
  data_aloe_m1vox.Stores.conv / data_aloe_v2vox.Stores.conv -> rowcache_vox.row_audio (bit-identical cache of) audio_s4q.ffmpeg_row_audio:
    ffmpeg -f f32le -ac 1 -ar 24000 of the WHOLE FLAC (WLT 'gcs' track or staged 'mimidec' Mimi decode)  ->  samples
    [1920 audio_frame0, 1920 (audio_frame0 + n_frames)) zero-padded  ->  x 10^(gain_db / 20) (recorded Mimi LUFS gain; mimidec 0 dB).
    NO clipping / limiting / normalisation anywhere after the gain.
  train_aloe_*.encode -> audio_s4q.VoxFront.encode -> voxtral_front.VoxtralFront.encode: crop/pad to 1920 n, resample 24 k -> 16 kHz
    (voxtral_front.resample, torchaudio sinc_interp_hann), crop to 1280 n, log-mel (128 bins), encoder.
  => measured twice: 'w24' = the waveform handed to Voxtral (front.encode input) and 'w16' = the 16 kHz waveform its log-mel is computed from.
  (mimidec FLACs were written by tools/stage_m1vox.py clipped to [-1, 1 - 2^-23] as PCM_24; that is part of the file, reproduced by reading it.)

Population: every TRAIN item (store, row, swap) of each plan (m1vox: data_aloe.make_plan(64) = 23,400 items; v2vox: data_aloe_v2.make_plan(64)
= 16,134 items, a subset of m1vox's: same (k, r, swap) -> same index entry). Dummy fallback micros and eval items excluded. The m1vox item set
is measured in full (covers v2vox); files are processed in a seeded random order, so any prefix is a uniform random cluster sample.

Metrics per item (on w24 and on w16): BS.1770-4 integrated loudness (pyloudnorm, as upstream's Mimi gain); RMS dBFS (20 log10 rms, full-scale
square = 0 dBFS) over the whole clip and over voiced 20 ms frames only (frame RMS > -50 dBFS); sample peak; fraction / count of samples with
|x| >= 0.995 (and >= 1.0).

usage (slinky-4, venv qwen-s2; GOOGLE_APPLICATION_CREDENTIALS = key file path):
  python -I loudness_audit.py plan    --out DIR
  python -I loudness_audit.py measure --out DIR --tmp /scratch/amartyavarma/loudness_audit_tmp --workers 44 --dl 16 [--max-files N]
  python -I loudness_audit.py report  --out DIR"""
import argparse
import collections
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import os
import sys
import threading
import time

import numpy as np

M1 = "/data/home/amartyavarma/ppm/aloe-m1vox-ft"
V2 = "/data/home/amartyavarma/ppm/aloe-v2vox-ft"
INDEX = f"{M1}/plan/M1VOX_INDEX.json"
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
BUCKET, PREFIX = "aloe-processed-datasets", "gs://aloe-processed-datasets/"
WORLD = 64
REF_DIGEST = {"m1vox": "f4d30485a70f55af", "v2vox": "ba2d70729ea437ab"}       # QQA_REF_PLAN_DIGEST of aloe_m1vox.env / aloe_v2vox.env
SEED = 20261017
VOICED_DB, FRAME_S, CLIP = -50.0, 0.020, 0.995
MIMIDEC_TMP = f"{M1}/loudness_audit/_tmp_mimidec"


def _paths():
    for p in (f"{M1}/code/lib", f"{M1}/code"):
        if p not in sys.path:
            sys.path.insert(0, p)


# ------------------------------------------------------------------------------------------------ plan
def plan(a):
    _paths()
    import pyarrow.parquet as pq
    for f in ("data_ppm.py", "data_qqa.py", "data_aloe.py"):               # v2vox's shared data code == m1vox's (as its trainer asserts)
        assert open(f"{M1}/code/{f}", "rb").read() == open(f"{V2}/code/{f}", "rb").read(), f
    for f in ("audio_s4q.py", "voxtral_front.py"):
        assert open(f"{M1}/code/lib/{f}", "rb").read() == open(f"{V2}/code/lib/{f}", "rb").read(), f
    import data_aloe as AL
    sys.path.insert(0, f"{V2}/code")
    import data_aloe_v2 as A2
    assert A2.SRC_STORES == AL.SRC_STORES and A2.DIGESTS == AL.DIGESTS
    PL = {"m1vox": AL.make_plan(WORLD, roots=AL.SRC_STORES), "v2vox": A2.make_plan(WORLD, roots=A2.SRC_STORES)}
    for run, P in PL.items():
        dg = hashlib.sha256(json.dumps(P["items"]).encode()).hexdigest()[:16]
        assert dg == REF_DIGEST[run], (run, dg, REF_DIGEST[run])
        print(f"[plan] {run}: digest {dg} == logged; steps {P['steps']} train items {len(P['items'])} eval {len(P['eval'])}", flush=True)
    assert len(PL["m1vox"]["items"]) == 23400 and len(PL["v2vox"]["items"]) == 16134
    m1 = [tuple(map(int, x)) for x in PL["m1vox"]["items"]]
    v2 = {tuple(map(int, x)) for x in PL["v2vox"]["items"]}
    assert len(set(m1)) == len(m1) and v2 <= set(m1), "v2vox items must be a subset of m1vox's"
    j = json.load(open(INDEX))
    assert j["digests"] == AL.DIGESTS, "index built for other stores"
    idx = {int(k): v for k, v in j["rows"].items()}
    cols = ["row", "row_key", "frame_start", "n_frames", "user_speaker_dir", "assistant_speaker_dir", "chunk_index"]
    rows = {k: pq.read_table(f"{AL.SRC_STORES[k]}/rows.parquet", columns=cols).to_pylist() for k in range(4)}
    man = {(r["session_id"], r["seg"]): r["relative_dir"] for r in csv.DictReader(open(f"{DATA}/manifest.csv"))}
    mj_cache = {}

    def mimi_lufs(ses, seg, spk):
        d = man.get((ses, seg))
        if d is None:
            return {}
        if d not in mj_cache:
            mj_cache[d] = json.load(open(f"{DATA}/{d}/w00.mimi.json"))["mimi"]
        m = mj_cache[d][spk]
        return {"lufs_in": m.get("lufs_in"), "gain_intended_db": m.get("lufs_gain_db_intended"), "mimi_gain_db": m.get("lufs_gain_db"),
                "mimi_status": m.get("lufs_status")}

    out = []
    for i, (k, r, sw) in enumerate(m1):
        row = rows[k][r]
        aside = "assistant" if sw else "user"                                # = data_aloe_m1vox.Stores.conv
        e = idx[k][r]
        assert e["row_key"] == row["row_key"] and e["row"] == r and row["row"] == r
        A = e["A"][aside]
        assert int(A["n_frames"]) == int(row["n_frames"]) and int(A["frame0"]) == int(row["frame_start"]), (k, r, sw)
        assert A["a_spk"] == row[f"{aside}_speaker_dir"]
        rec = {"i": i, "item": [k, r, sw], "in_v2vox": int((k, r, sw) in v2), "row_key": row["row_key"], "side": aside, "spk": A["a_spk"],
               "chunk": int(row["chunk_index"]), "kind": A["kind"], "flac": A["flac"], "gain_db": float(A["gain_db"]),
               "audio_frame0": int(A["audio_frame0"]), "n_frames": int(A["n_frames"]), "lufs_status": A.get("lufs_status")}
        if (k, r, sw) in v2:
            assert int(row["chunk_index"]) == 0
        if A["kind"] == "gcs":
            assert A["uri"] == PREFIX + "stage2-handover-20261008/wlt/" + A["flac"]
            rec.update(uri=A["uri"], generation=int(A["generation"]), bytes=int(A["flac_bytes"]))
            ses, seg, _ = A["flac"].split("/")
            ml = mimi_lufs(ses, seg, A["a_spk"])
            assert abs(float(ml["mimi_gain_db"]) - rec["gain_db"]) < 1e-9, (rec, ml)
            rec.update(ml)
        else:
            assert A["kind"] == "mimidec" and rec["gain_db"] == 0.0 and rec["audio_frame0"] == 0
        out.append(rec)
    files = sorted({r["flac"] for r in out})
    perm = np.random.default_rng(SEED).permutation(len(files))
    order = [files[p] for p in perm]
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/items.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    json.dump(order, open(f"{a.out}/file_order.json", "w"))
    kinds = collections.Counter(r["kind"] for r in out)
    gb = sum({r["flac"]: r.get("bytes", 0) for r in out}.values()) / 1e9
    print(f"[plan] {len(out)} m1vox train items ({sum(r['in_v2vox'] for r in out)} also v2vox); kinds {dict(kinds)}; distinct files {len(files)} "
          f"(gcs {sum(1 for x in files if not x.startswith('_mimidec'))}, {gb:.1f} GB); status {dict(collections.Counter(r['lufs_status'] for r in out))}",
          flush=True)


# ------------------------------------------------------------------------------------------------ measure
def metrics(x, sr):
    import pyloudnorm as pyln
    x = np.asarray(x, np.float32)
    ax = np.abs(x)
    fl = int(round(FRAME_S * sr))
    nfr = len(x) // fl
    ms = (x[:nfr * fl].astype(np.float64).reshape(nfr, fl) ** 2).mean(1)
    fdb = 10.0 * np.log10(ms + 1e-30)
    v = fdb > VOICED_DB
    lufs = float(pyln.Meter(sr).integrated_loudness(x.astype(np.float64)))
    pk = float(ax.max())
    return {"lufs": lufs if math.isfinite(lufs) else None,
            "rms_db": float(10.0 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-30)),
            "vrms_db": float(10.0 * np.log10(ms[v].mean())) if v.any() else None,
            "voiced_frac": float(v.mean()),
            "peak": pk, "peak_db": float(20.0 * np.log10(pk + 1e-30)),
            "n_clip": int((ax >= CLIP).sum()), "clip_frac": float((ax >= CLIP).mean()), "n_ge1": int((ax >= 1.0).sum())}


_W = {}


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _paths()
    import torch
    torch.set_num_threads(1)
    import audio_s4q as AU
    import voxtral_front as VF
    _W["AU"], _W["VF"], _W["torch"] = AU, VF, torch
    orig = AU.ffmpeg_decode_24k
    cache = {}

    def memo(path):                       # one ffmpeg decode per FLAC for all its rows (ffmpeg_row_audio's own slice / pad / gain run per row)
        if path not in cache:
            cache.clear()
            cache[path] = orig(path)
        return cache[path]
    AU.ffmpeg_decode_24k = memo


def work(path, recs):
    AU, VF, torch = _W["AU"], _W["VF"], _W["torch"]
    out = []
    for r in recs:
        t0 = time.time()
        w24 = AU.ffmpeg_row_audio({"flac": path, "frame0": r["audio_frame0"], "n_frames": r["n_frames"], "row_key": r["row_key"]}, r["gain_db"])
        n = r["n_frames"]
        assert w24.dtype == np.float32 and w24.shape == (n * AU.SAMPLES_PER_FRAME_24K,)
        w16 = VF.resample(torch.from_numpy(w24)[None], VF.SR_24K, VF.SR_16K)[..., : n * VF.SAMPLES_PER_FRAME_16K][0].numpy()
        assert w16.shape == (n * VF.SAMPLES_PER_FRAME_16K,)
        out.append({"i": r["i"], "w24": metrics(w24, 24000), "w16": metrics(w16, 16000), "dt": round(time.time() - t0, 2)})
    return out


def measure(a):
    items = [json.loads(x) for x in open(f"{a.out}/items.jsonl")]
    order = json.load(open(f"{a.out}/file_order.json"))
    by = collections.defaultdict(list)
    for r in items:
        by[r["flac"]].append(r)
    res_p = f"{a.out}/results.jsonl"
    done = set()
    if os.path.exists(res_p):
        for x in open(res_p):
            done.add(json.loads(x)["i"])
    todo = [f for f in order if any(r["i"] not in done for r in by[f])]
    if a.max_files:
        todo = todo[:a.max_files]
    mimi_todo = [f for f in todo if by[f][0]["kind"] == "mimidec"]
    gcs_todo = [f for f in todo if by[f][0]["kind"] == "gcs"]
    if not a.with_mimidec:
        mimi_todo = []
    missing = [f for f in mimi_todo if not os.path.exists(f"{MIMIDEC_TMP}/{f}")]
    assert not missing, ("mimidec copies missing (fetch them first)", len(missing), missing[:3])
    if a.only_mimidec:
        gcs_todo = []
    print(f"[measure] {len(done)} items already done; now {len(gcs_todo)} gcs files + {len(mimi_todo)} mimidec files", flush=True)
    from google.cloud import storage
    cli = storage.Client.from_service_account_json(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    bkt = cli.bucket(BUCKET)
    os.makedirs(a.tmp, exist_ok=True)
    sem = threading.Semaphore(a.ahead)
    lock = threading.Lock()
    st = {"files": 0, "items": 0, "bytes": 0, "t0": time.time()}
    fout = open(res_p, "a")

    def fetch(rel):
        r0 = by[rel][0]
        dst = f"{a.tmp}/{rel.replace('/', '__')}"
        blob = bkt.blob(r0["uri"][len(PREFIX):], generation=r0["generation"])
        for att in range(6):
            try:
                blob.download_to_filename(dst, checksum="crc32c")          # crc32c validated by the client
                break
            except Exception as e:  # noqa: BLE001
                if att == 5:
                    raise
                print(f"[retry {att + 1}] {rel}: {e!r}", flush=True)
                time.sleep(5 * (att + 1))
        assert os.path.getsize(dst) == r0["bytes"], (rel, os.path.getsize(dst), r0["bytes"])
        return dst

    import multiprocessing as mp
    with cf.ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as pool, cf.ThreadPoolExecutor(a.dl) as dl:
        futs = {}

        def on_done(fut, rel, path, delete):
            try:
                out = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] {rel}: {e!r}", flush=True)
                out = []
            finally:
                if delete and os.path.exists(path):
                    os.remove(path)
                sem.release()
            with lock:
                for o in out:
                    fout.write(json.dumps(o) + "\n")
                fout.flush()
                st["files"] += 1; st["items"] += len(out); st["bytes"] += by[rel][0].get("bytes", 0)
                if st["files"] % 200 == 0:
                    el = time.time() - st["t0"]
                    print(f"[measure] {st['files']}/{len(gcs_todo) + len(mimi_todo)} files, {st['items']} items, {st['bytes'] / 1e9:.1f} GB "
                          f"({st['bytes'] / 1e6 / el:.0f} MB/s) {el:.0f}s", flush=True)

        def launch(rel):
            recs = [r for r in by[rel] if r["i"] not in done]
            try:
                if by[rel][0]["kind"] == "gcs":
                    path, delete = fetch(rel), True
                else:
                    path, delete = f"{MIMIDEC_TMP}/{rel}", False
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR fetch] {rel}: {e!r}", flush=True)
                sem.release()
                return None
            f = pool.submit(work, path, recs)
            f.add_done_callback(lambda fut, rel=rel, path=path, delete=delete: on_done(fut, rel, path, delete))
            return f

        subs = []
        for rel in mimi_todo + gcs_todo:
            sem.acquire()
            subs.append(dl.submit(launch, rel))
        pf = [s.result() for s in subs]
        cf.wait([f for f in pf if f is not None])
    fout.close()
    print(f"[measure] DONE {st['files']} files {st['items']} items in {time.time() - st['t0']:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------ report
KEYS = [("lufs", "integrated LUFS"), ("vrms_db", "voiced RMS dBFS"), ("rms_db", "whole-clip RMS dBFS"), ("peak", "sample peak"),
        ("peak_db", "sample peak dBFS"), ("clip_frac", "frac |x|>=0.995"), ("voiced_frac", "voiced-frame frac")]


def stats(v):
    v = np.asarray([x for x in v if x is not None and math.isfinite(x)], np.float64)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "min": float(v.min()),
            "p5": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def report(a):
    items = {r["i"]: r for r in (json.loads(x) for x in open(f"{a.out}/items.jsonl"))}
    res = {}
    for x in open(f"{a.out}/results.jsonl"):
        o = json.loads(x)
        res[o["i"]] = o
    pops = {"m1vox": [i for i in items], "v2vox": [i for i in items if items[i]["in_v2vox"]]}
    S = {"population": {k: len(v) for k, v in pops.items()}, "measured": {}, "runs": {}}
    lines = []
    for run, ids in pops.items():
        got = [i for i in ids if i in res]
        S["measured"][run] = len(got)
        groups = {"all": got, "gcs": [i for i in got if items[i]["kind"] == "gcs"], "mimidec": [i for i in got if items[i]["kind"] == "mimidec"],
                  "gcs status=ok": [i for i in got if items[i]["kind"] == "gcs" and items[i]["lufs_status"] == "ok"],
                  "gcs status=ok+peak_limited": [i for i in got if items[i]["kind"] == "gcs" and items[i]["lufs_status"] == "ok+peak_limited"]}
        S["runs"][run] = {}
        for g, gi in groups.items():
            S["runs"][run][g] = {}
            for wk in ("w24", "w16"):
                S["runs"][run][g][wk] = {k: stats([res[i][wk][k] for i in gi]) for k, _ in KEYS}
                c = [res[i][wk] for i in gi]
                S["runs"][run][g][wk]["rows_any_clip"] = int(sum(1 for o in c if o["n_clip"] > 0))
                S["runs"][run][g][wk]["rows_any_ge1"] = int(sum(1 for o in c if o["n_ge1"] > 0))
                S["runs"][run][g][wk]["rows_lufs_undefined"] = int(sum(1 for o in c if o["lufs"] is None))
                S["runs"][run][g][wk]["n_clip_total"] = int(sum(o["n_clip"] for o in c))
            S["runs"][run][g]["gain_db"] = stats([items[i]["gain_db"] for i in gi])
            S["runs"][run][g]["n"] = len(gi)
        L = np.asarray([res[i]["w24"]["lufs"] for i in got if res[i]["w24"]["lufs"] is not None])
        S["runs"][run]["near_target"] = {f"within_{t}LU": float(np.mean(np.abs(L + 23.0) <= t)) for t in (0.5, 1.0, 3.0)}
        S["runs"][run]["near_target"].update({"below_-26": float(np.mean(L < -26)), "below_-30": float(np.mean(L < -30)), "above_-20": float(np.mean(L > -20))})
        lines.append(f"== {run}: share of rows by w24 LUFS vs -23: {json.dumps({k: round(v, 4) for k, v in S['runs'][run]['near_target'].items()})}")
        lines.append(f"== {run}: {len(got)} of {len(ids)} train row-sides measured (gcs {len(groups['gcs'])}, mimidec {len(groups['mimidec'])}; "
                     f"status ok {len(groups['gcs status=ok'])}, ok+peak_limited {len(groups['gcs status=ok+peak_limited'])})")
        for g in groups:
            for wk, lab in (("w24", "24 kHz waveform into Voxtral front.encode"), ("w16", "16 kHz waveform the log-mel is computed from")):
                if g != "all" and wk == "w16":
                    continue
                d = S["runs"][run][g][wk]
                lines.append(f"  [{g}] {lab}: rows with any |x|>=0.995: {d['rows_any_clip']}, any |x|>=1: {d['rows_any_ge1']}, "
                             f"LUFS undefined (silent): {d['rows_lufs_undefined']}")
                lines.append(f"    {'metric':22s} {'n':>6s} {'mean':>9s} {'std':>8s} {'min':>9s} {'p5':>9s} {'p50':>9s} {'p95':>9s} {'max':>9s}")
                for k, lab2 in KEYS:
                    s = d[k]
                    if not s.get("n"):
                        continue
                    fmt = "{:9.2e}" if k == "clip_frac" else ("{:9.4f}" if k in ("peak", "voiced_frac") else "{:9.2f}")
                    lines.append(f"    {lab2:22s} {s['n']:6d} {s['mean']:9.3f} {s['std']:8.3f} " + " ".join(fmt.format(s[q]) for q in ("min", "p5", "p50", "p95", "max")))
            s = S["runs"][run][g]["gain_db"]
            if s.get("n"):
                lines.append(f"    {'applied gain dB':22s} {s['n']:6d} {s['mean']:9.3f} {s['std']:8.3f} " + " ".join(f"{s[q]:9.2f}" for q in ("min", "p5", "p50", "p95", "max")))
    # extremes (m1vox, w24)
    got = [i for i in pops["m1vox"] if i in res]
    q = sorted([i for i in got if res[i]["w24"]["lufs"] is not None], key=lambda i: res[i]["w24"]["lufs"])
    S["quietest"] = [{"row_key": items[i]["row_key"], "side": items[i]["side"], "kind": items[i]["kind"], "lufs": res[i]["w24"]["lufs"],
                      "vrms_db": res[i]["w24"]["vrms_db"], "voiced_frac": res[i]["w24"]["voiced_frac"], "gain_db": items[i]["gain_db"],
                      "status": items[i]["lufs_status"]} for i in q[:15]]
    S["loudest"] = [{"row_key": items[i]["row_key"], "side": items[i]["side"], "kind": items[i]["kind"], "lufs": res[i]["w24"]["lufs"],
                     "peak": res[i]["w24"]["peak"], "gain_db": items[i]["gain_db"]} for i in q[-10:]]
    S["silent_rows"] = [{"row_key": items[i]["row_key"], "side": items[i]["side"], "kind": items[i]["kind"], "voiced_frac": res[i]["w24"]["voiced_frac"]}
                        for i in got if res[i]["w24"]["lufs"] is None]
    S["w24_minus_w16_lufs"] = stats([res[i]["w24"]["lufs"] - res[i]["w16"]["lufs"] for i in got
                                     if res[i]["w24"]["lufs"] is not None and res[i]["w16"]["lufs"] is not None])
    lines.append(f"w24 - w16 integrated LUFS: {S['w24_minus_w16_lufs']}")
    # pipeline check vs upstream: the episode-level loudness after the recorded gain = lufs_in + gain (-23 for 'ok', lower when peak-limited);
    # a row that IS the whole episode (single-chunk conversation) must measure exactly that
    _paths()
    import pyarrow.parquet as pq
    import data_aloe as AL
    nch = {k: pq.read_table(f"{AL.SRC_STORES[k]}/rows.parquet", columns=["n_chunks_conv"]).column(0).to_pylist() for k in range(4)}
    ex = [i for i in got if items[i].get("lufs_in") is not None and res[i]["w24"]["lufs"] is not None]
    d_all = [res[i]["w24"]["lufs"] - (items[i]["lufs_in"] + items[i]["gain_db"]) for i in ex]
    one = [i for i in ex if int(nch[items[i]["item"][0]][items[i]["item"][1]]) == 1]
    d_one = [res[i]["w24"]["lufs"] - (items[i]["lufs_in"] + items[i]["gain_db"]) for i in one]
    S["episode_target_after_gain"] = stats([items[i]["lufs_in"] + items[i]["gain_db"] for i in ex])
    S["row_minus_episode_lufs"] = stats(d_all)
    S["row_minus_episode_lufs_single_chunk"] = stats(d_one)
    S["peak_limited_shortfall_db"] = stats([items[i]["gain_intended_db"] - items[i]["gain_db"] for i in ex if items[i]["lufs_status"] == "ok+peak_limited"])
    lines.append(f"episode loudness after recorded gain (lufs_in + gain, m1vox gcs rows): {S['episode_target_after_gain']}")
    lines.append(f"peak-limited rows: intended - applied gain dB: {S['peak_limited_shortfall_db']}")
    lines.append(f"row LUFS - episode LUFS after gain: all {S['row_minus_episode_lufs']}")
    lines.append(f"  single-chunk conversations (row = whole episode; pipeline check): {S['row_minus_episode_lufs_single_chunk']}")
    lines.append("quietest 15 (m1vox, w24): " + json.dumps(S["quietest"]))
    json.dump(S, open(f"{a.out}/summary.json", "w"), indent=1)
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "measure", "report"])
    ap.add_argument("--out", default=f"{M1}/loudness_audit")
    ap.add_argument("--tmp", default="/scratch/amartyavarma/loudness_audit_tmp")
    ap.add_argument("--workers", type=int, default=44)
    ap.add_argument("--dl", type=int, default=16)
    ap.add_argument("--ahead", type=int, default=160)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--with-mimidec", type=int, default=1)
    ap.add_argument("--only-mimidec", action="store_true")
    a = ap.parse_args()
    {"plan": plan, "measure": measure, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
