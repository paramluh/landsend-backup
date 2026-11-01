"""v2pure: copy of scripts/extract.py reading ONLY gs://aloe-processed-data/pose-speech-vad/releases/v2/ (selector = the release's own
manifest.parquet, rows with eligible == true AND is_active_speaker == true; user's active-speaker rule). Changes vs extract.py: v2 paths,
done.json files are a list with crc32c (not sha256), v2 metadata keys (Mimi checkpoint revision under mimi.recipe, attention policy
checkpoint_sliding_window_v1; pose tokenizer repo/revision/recipe asserted as before), listening rows not extracted (the SAM-active builder
drops them first anyway). Output schema identical to raw/recordings.

ORIGINAL DOC: Aloe pose-speech-vad -> compact token-only raw copy on Weka (CLEAN selection only).

For each recording in manifests/clean-20261005-parquet-v1/clean.parquet: download its 4 parquet files + done.json from
gs://aloe-processed-data/pose-speech-vad/recordings/<id>/ INTO MEMORY, verify bytes + sha256 against done.json, read the clean
speaker-segments' row groups (asserting recording / speaker / segment id against the selector), and write ONE zstd parquet per recording
to raw/recordings/<id>.parquet (one row per clean speaker-segment) with everything except the per-frame `confidence` array and the
duplicated `words_json` documents (both stay in the bucket). Per-recording stats -> raw/stats/<id>.json.

usage: python -I extract.py --part K --nparts N --workers W
Credentials: service-account key by path (never printed).
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import traceback
from collections import Counter

import numpy as np

KEY = "/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json"
BUCKET, PREFIX = "aloe-processed-data", "pose-speech-vad/releases/v2/recordings/"
ROOT = "/data/home/amartyavarma/aloe-pose"
SEL = f"{ROOT}/meta-v2/manifest.parquet"
OUT = f"{ROOT}/raw-v2pure/recordings"
STATS = f"{ROOT}/raw-v2pure/stats"
MIN_FREE_GB = 310
POLICY = "checkpoint_sliding_window_v1"
_CL = [None]


def bucket():
    if _CL[0] is None:
        from google.cloud import storage
        _CL[0] = storage.Client.from_service_account_json(KEY).bucket(BUCKET)
    return _CL[0]


def schema():
    import pyarrow as pa
    i16 = pa.list_(pa.list_(pa.int16()))
    f32 = pa.list_(pa.list_(pa.float32()))
    wt = pa.list_(pa.struct([("word", pa.string()), ("start", pa.float64()), ("end", pa.float64()), ("local_start", pa.float64()),
                             ("local_end", pa.float64()), ("crosses_boundary", pa.bool_())]))
    return pa.schema([
        ("recording_id", pa.string()), ("segment_id", pa.string()), ("segment_index", pa.int32()), ("speaker", pa.string()),
        ("active_speaker", pa.string()), ("is_active_speaker", pa.bool_()), ("partner_eligible", pa.bool_()),
        ("audio_crossbleed_status", pa.string()), ("transcript_status", pa.string()),
        ("start", pa.float64()), ("end", pa.float64()), ("requested_start", pa.float64()), ("requested_end", pa.float64()),
        ("start_frame", pa.int64()), ("num_frames", pa.int64()), ("n_codes", pa.int32()), ("sample_start", pa.int64()), ("sample_end", pa.int64()),
        ("pose_tokens", i16), ("mimi_tokens", i16), ("frame0_pose", f32), ("global_state0", pa.list_(pa.float32())), ("global_state_const", pa.bool_()),
        ("orig_h", pa.float32()), ("orig_w", pa.float32()), ("conf_mean55", pa.list_(pa.float32())),
        ("lufs_status", pa.string()), ("lufs_target", pa.float32()), ("lufs_in", pa.float32()),
        ("words", wt), ("pose_metadata_json", pa.string()), ("audio_metadata_json", pa.string()), ("words_metadata_json", pa.string()),
        ("segment_metadata_json", pa.string()), ("original_pose_sha256", pa.string()), ("original_mimi_sha256", pa.string()),
        ("original_words_sha256", pa.string())])


def one(job):
    rid, segs = job                                         # segs: list of selector dicts for this recording
    import pyarrow as pa
    import pyarrow.parquet as pq
    dst = f"{OUT}/{rid}.parquet"
    if os.path.exists(dst) and os.path.exists(f"{STATS}/{rid}.json"):
        return rid, "skip", None
    while True:
        sv = os.statvfs(ROOT)
        if sv.f_bavail * sv.f_frsize >= MIN_FREE_GB * 2**30:
            break
        time.sleep(60)
    t0 = time.time(); st = Counter()
    b = bucket()
    done = json.loads(b.blob(f"{PREFIX}{rid}/done.json").download_as_bytes())
    assert done.get("recording_id") == rid and done.get("state") == "completed", (rid, "done.json")
    raw = {}
    import base64, google_crc32c
    for meta in done["files"]:
        f = meta["path"].split(f"/recordings/{rid}/", 1)[1]
        x = b.blob(f"{PREFIX}{rid}/{f}").download_as_bytes()
        assert len(x) == meta["bytes"], (rid, f, len(x), meta["bytes"])
        assert base64.b64encode(google_crc32c.value(x).to_bytes(4, "big")).decode() == meta["crc32c"], (rid, f, "crc32c")
        raw[f] = x; st["bytes_downloaded"] += len(x)
    pfs = {f: pq.ParquetFile(io.BytesIO(v)) for f, v in raw.items()}
    rows = []
    for s in sorted(segs, key=lambda s: (s["speaker"], s["row_group_index"])):
        sp, rg = s["speaker"], int(s["row_group_index"])
        pr = pfs[f"{sp}/pose.parquet"].read_row_group(rg).to_pylist()
        ar = pfs[f"{sp}/audio.parquet"].read_row_group(rg).to_pylist()
        assert len(pr) == 1 and len(ar) == 1, (rid, sp, rg)
        p, a = pr[0], ar[0]
        for r in (p, a):
            assert r["recording_id"] == rid and r["speaker"] == sp and r["segment_id"] == s["segment_id"], (rid, sp, rg, r["segment_id"], s["segment_id"])
            assert int(r["start_frame"]) == int(s["start_frame"]) and int(r["start_frame"]) + int(r["num_frames"]) == int(s["end_frame"]), (rid, sp, rg)
        assert p["active_speaker"] == s["active_speaker"] == a["active_speaker"]
        nf = int(p["num_frames"]); assert nf % 2 == 0 and nf == int(a["num_frames"])
        n = nf // 2
        pt = np.asarray(p["pose_tokens"], np.int16); mt = np.asarray(a["mimi_tokens"], np.int16)
        assert pt.shape == (32, n) and mt.shape == (32, n), (rid, sp, rg, pt.shape, mt.shape, n)
        assert pt.min() >= 0 and pt.max() < 2048 and mt.min() >= 0 and mt.max() < 2048, (rid, sp, rg, "code range")
        f0 = np.asarray(p["frame0_pose"], np.float32); assert f0.shape == (133, 3), f0.shape
        gs = np.asarray(p["global_state"], np.float32); assert gs.shape == (nf, 3), (gs.shape, nf)
        gconst = bool((gs == gs[0]).all())
        conf = np.asarray(p["confidence"], np.float32); assert conf.shape == (nf, 55), conf.shape
        pm = json.loads(p["metadata_json"]); am = json.loads(a["metadata_json"])
        pz = pm["pose"]
        assert pz["revision"] == "1d6ff4f0de89408cb054da618e309d4334596252" and pz["repo"] == "InternalCan/pose-tokenizer-32cb-200k-noshift"
        assert pz["normalization_recipe"] == "frame0-root-frame0-horizontal-shoulder-width-v1" and pz.get("token_time_anchor", "end_of_two_frame_group") == "end_of_two_frame_group"
        assert abs(float(pz.get("token_stride_s", pz["model_contract"]["token_stride_s"])) - 0.08) < 1e-9 and pm["code_rate_hz"] == 12.5
        assert pz["coordinate_preprocessing"]["name"] == "confidence-hold-03-frame0-init-v2", (rid, sp, rg, "pose version")
        mm = am["mimi"]; mr = mm["recipe"]
        assert mr["checkpoint_revision"] == "89091b3e466eb6a9d11e537bf26b144f194978f7" and int(mr["codebooks"]) == 32 and int(mm["code_count"]) == n
        assert mm["attention_policy"] == POLICY == mr["attention_policy"], (rid, sp, rg, "mimi attention policy")
        norm = mm.get("normalization") or {}
        st["lufs_" + str(norm.get("lufs_status"))] += 1
        st["gs_nonconst"] += int(not gconst); st["segments"] += 1; st["codes"] += n
        st["frame0_invalid"] += int(not pz.get("frame0_source_valid", True))
        words = [{k: w[k] for k in ("word", "start", "end", "local_start", "local_end", "crosses_boundary")} for w in (a["words"] or [])]
        st["words"] += len(words); st["words_crossing"] += sum(1 for w in words if w["crosses_boundary"])
        rows.append({"recording_id": rid, "segment_id": s["segment_id"], "segment_index": int(p["segment_index"]), "speaker": sp,
                     "active_speaker": p["active_speaker"], "is_active_speaker": bool(s["is_active_speaker"]), "partner_eligible": bool(s["partner_eligible"]),
                     "audio_crossbleed_status": s["audio_crossbleed_status"], "transcript_status": s["transcript_status"],
                     "start": p["start"], "end": p["end"], "requested_start": p["requested_start"], "requested_end": p["requested_end"],
                     "start_frame": int(p["start_frame"]), "num_frames": nf, "n_codes": n, "sample_start": int(p["sample_start"]), "sample_end": int(p["sample_end"]),
                     "pose_tokens": pt.tolist(), "mimi_tokens": mt.tolist(), "frame0_pose": f0.tolist(), "global_state0": gs[0].tolist(),
                     "global_state_const": gconst, "orig_h": float(pz["orig_h"]), "orig_w": float(pz["orig_w"]), "conf_mean55": conf.mean(0).tolist(),
                     "lufs_status": str(norm.get("lufs_status")), "lufs_target": float(norm.get("lufs_target", float("nan"))),
                     "lufs_in": float(norm["lufs_in"]) if norm.get("lufs_in") is not None else float("nan"),
                     "words": words, "pose_metadata_json": p["metadata_json"], "audio_metadata_json": a["metadata_json"],
                     "words_metadata_json": a.get("words_metadata_json"), "segment_metadata_json": p["segment_metadata_json"],
                     "original_pose_sha256": p.get("original_pose_sha256"), "original_mimi_sha256": a.get("original_mimi_sha256"),
                     "original_words_sha256": a.get("original_words_sha256")})
    tbl = pa.Table.from_pylist(rows, schema=schema())
    tmp = f"{dst}.tmp.{os.getpid()}"
    pq.write_table(tbl, tmp, compression="zstd", compression_level=6, row_group_size=64)
    os.replace(tmp, dst)
    st["bytes_written"] = os.path.getsize(dst); st["seconds"] = round(time.time() - t0, 2)
    js = {"recording_id": rid, "stats": dict(st), "done_json": done}
    with open(f"{STATS}/{rid}.json.tmp", "w") as f:
        json.dump(js, f)
    os.replace(f"{STATS}/{rid}.json.tmp", f"{STATS}/{rid}.json")
    return rid, "ok", dict(st)


def one_safe(job):
    for attempt in range(4):
        try:
            return one(job)
        except Exception as e:  # noqa: BLE001
            err = traceback.format_exc()
            time.sleep(5 * (attempt + 1))
    with open(f"{ROOT}/raw-v2pure/errors/{job[0]}.txt", "w") as f:
        f.write(err)
    return job[0], "fail", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    for d in (OUT, STATS, f"{ROOT}/raw-v2pure/errors"):
        os.makedirs(d, exist_ok=True)
    cols = ["recording_id", "segment_id", "speaker", "active_speaker", "is_active_speaker", "start_frame", "end_frame", "partner_eligible",
            "audio_crossbleed_status", "transcript_status", "row_group_index", "eligible"]
    t = pq.read_table(SEL, columns=cols).to_pylist()
    by = {}
    for r in t:
        if not (r["eligible"] and r["is_active_speaker"]):
            continue
        by.setdefault(r["recording_id"], []).append(r)
    rids = sorted(by)
    mine = rids[a.part::a.nparts]
    if a.limit:
        mine = mine[:a.limit]
    jobs = [(r, by[r]) for r in mine]
    print(f"[extract] part {a.part}/{a.nparts}: {len(jobs)} recordings, {sum(len(j[1]) for j in jobs)} segments, {a.workers} workers", flush=True)
    import multiprocessing as mp
    t0 = time.time(); n = 0; nb = 0; nfail = 0; nskip = 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=200) as pool:
        for rid, status, st in pool.imap_unordered(one_safe, jobs, chunksize=1):
            n += 1
            if status == "ok":
                nb += st["bytes_downloaded"]
            elif status == "fail":
                nfail += 1; print(f"[extract] FAIL {rid}", flush=True)
            else:
                nskip += 1
            if n % 100 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"[extract] {n}/{len(jobs)} skip {nskip} fail {nfail} | {nb / 1e9:.1f} GB in {el:.0f}s = {nb / 1e6 / max(el, 1):.0f} MB/s", flush=True)
    print(f"[extract] DONE part {a.part}: {n} recs, fail {nfail}", flush=True)


if __name__ == "__main__":
    main()
