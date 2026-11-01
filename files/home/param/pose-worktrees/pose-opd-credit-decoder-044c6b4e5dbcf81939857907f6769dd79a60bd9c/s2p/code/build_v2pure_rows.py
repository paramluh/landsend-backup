"""aloe-v2pure-ft DATA BUILD (10-17): the Aloe pose-speech-vad release v2 (= the 8B "SAM pose Aloe v2pure" fine-tune's data, ONLY this
source: the local cache /data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure, built from gs://aloe-processed-data/pose-speech-vad/
releases/v2 by aloe-pose/scripts-v2pure/build_sam_cache_v2pure.py) -> 8 s3-duplex-rows-v1 stores p0..p7 + the pose add-on, the exact layout
of aloe-pose-ft/stores (build_poseB_rows.py; read by code/data_aloe_pose.py Stores / splits on top of lib/s3rows.py):
  rows-00000-of-00001.{codes_user,codes_assistant}.npy, .index.npz, .words.arrow, .persona.arrow, .DONE, rows.jsonl, rows.parquet, NODE.json
  (Model 1's s2h_to_s3rows.write_node, unchanged) + rows-00000-of-00001.pose_{user,assistant}.npy int16 [R, 32, 7500], pose_index.npz
  (b_ok [R, 2] uint8 (user, assistant), split [R] int8 0 train / 1 eval), POSE.json, BUILD_DONE; STORES.json + PLAN.json at the root.

ROW DESIGN (user decision 10-17):
  ONE row per cache item (one active-speaker clip, n = the clip's frames, 1..7494; the trainer runs with S3_MIN_FRAMES=1).
  row_key ALOE/aloe-v2pure/<rid>/c<k:05d>, k = the clip's ordinal in its recording (time order by clip start, ties by (speaker, segment_id),
  over the recording's clips of both speakers); chunk_index k, frame_start 7500 k, frame_end 7500 k + n (s3rows' rule); episode_id rid,
  podcast_id 'aloe-v2pure', dataset 'ALOE'. sysprompt.rec_key(row_key) == ALOE/aloe-v2pure/<rid> (asserted): one recording = one voice-clip
  sibling group, and every clip of a recording lies in ONE store (the sibling index is per store).
  Sides: speaker_1 = data 'assistant', speaker_2 = data 'user' on every row (assistant_speaker <rid>/speaker_1, user_speaker <rid>/speaker_2).
  Channel B = the clip's active speaker: that side carries the clip's Mimi codes, pose codes, words; pose_index b_ok = 1 on that side only.
  The partner side is ABSENT in this data (the trainer runs QQA_A_EMPTY=1: A never reaches the model) but check_row needs valid codes on
  both sides: partner codes = Mimi codes of digital silence (SIL_CODES 'seq' = the steady-state frame sequence Mimi emits for 24 kHz zeros,
  tiled from the clip's frame 0: the PPM trainer's xdep 'silent' / infer_ppm sil_seq / s3probe / Demeter silence codes), partner words empty,
  partner pose -1; provenance a_present False, partner 'absent'.
  Words: the cache item's words ([start, end, text], clip-local seconds, taken by the cache builder by word midpoint, so a first word can
  start before the clip): start in [-0.5, 0) clamped to 0, start < -0.5 dropped (the pose-s3 / tt-s3 rule), then Model 1's side_words
  unchanged (t0 0, dur n / 12.5: empty / CJK dropped, end clamped to the clip, s3rows frame rule, sorted); pieces = speech_q.word_pieces on
  Model 1's tokenizer (Qwen3-32B). Every drop / clamp counted.
  Persona sidecar: every row 'missing' on both sides (the release has no persona), written by write_node's sidecar path.
SPLIT: eval iff blake2b8('aloe-pose-v1|' + recording_id) (little-endian) % 10000 < 100 = the cache builder's is_eval (imported and compared
  on every recording) = the cache's own eval shard (asserted). STORES: recordings assigned by LPT on frames, eval recordings first, then
  train recordings (frames balanced per split across stores); rows in a store ordered by (rid, k).
usage (compute nodes only):
  python -I build_v2pure_rows.py plan  --out DIR --stores 8 --workers 32
  python -I build_v2pure_rows.py build --out DIR --store K --workers 8      (resumable: skips a store with BUILD_DONE; worker results cached
                                                                             under /scratch/amartyavarma/aloe-v2pure-build/pK, deleted after)
  python -I build_v2pure_rows.py stores --out DIR                          (STORES.json once every store has BUILD_DONE)"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

os.environ.setdefault("S3_MIN_FRAMES", "1")       # one row per clip: clips are 1..7494 frames (s3rows.check_row's floor)
assert os.environ["S3_MIN_FRAMES"] == "1", os.environ["S3_MIN_FRAMES"]

import numpy as np  # noqa: E402

sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch-data/s2h-m1")
import s2h_to_s3rows as B  # noqa: E402   (Model 1's converter: s3rows F, side_words, tok, write_node + the persona sidecar)

import sysprompt as SPR  # noqa: E402   (qwen-1ch/stage3, on B's path; rec_key identical to code/lib/sysprompt.py)

F = B.F
assert F.MIN_FRAMES == 1, F.MIN_FRAMES
CACHE = "/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure"
CACHE_BUILDER = "/data/home/amartyavarma/aloe-pose/scripts-v2pure/build_sam_cache_v2pure.py"
SIL_CODES = "/data/home/amartyavarma/setup-logs/s3hestia/ourrows/silence_codes.json"
TMP_ROOT = "/scratch/amartyavarma/aloe-v2pure-build"
TOKENIZER_POSE = "InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0de89408cb054da618e309d4334596252"
NORM = "frame0-root-frame0-horizontal-shoulder-width-v1"
RECIPE = "confidence-hold-03-frame0-init-v2"
EVAL_SALT, EVAL_MOD, EVAL_LT = "aloe-pose-v1|", 10000, 100
EXPECT = {"train": {"items": 367496, "frames": 92132615, "recordings": 12384}, "eval": {"items": 3482, "frames": 868932, "recordings": 113}}
SIDE_OF = {"speaker_1": "assistant", "speaker_2": "user"}
SOURCE = "gs://aloe-processed-data/pose-speech-vad/releases/v2 (local sam-cache-active-v2pure: active-speaker clips)"
B.DATASET, B.PODCAST = "ALOE", "aloe-v2pure"
B.VERSION = "aloe-v2pure-ft-v1"
ITEM_COLS = ["key", "rid", "speaker", "segment_id", "n", "start", "end", "words", "is_active_speaker", "active_speaker",
             "audio_crossbleed_status", "lufs_status"]


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def is_eval(rid):
    return h64(EVAL_SALT + rid) % EVAL_MOD < EVAL_LT


def cache_builder():
    spec = importlib.util.spec_from_file_location("build_sam_cache_v2pure", CACHE_BUILDER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert (m.EVAL_SALT, m.EVAL_MOD, m.EVAL_LT) == (EVAL_SALT, EVAL_MOD, EVAL_LT), (m.EVAL_SALT, m.EVAL_MOD, m.EVAL_LT)
    return m


_SIL = [None]


def sil_seq():
    if _SIL[0] is None:
        q = np.asarray(json.load(open(SIL_CODES))["seq"], np.int16)
        assert q.ndim == 2 and q.shape[0] == F.DEPTH and int(q.min()) >= 0 and int(q.max()) < F.CBS, q.shape
        _SIL[0] = q
    return _SIL[0]


def sil_codes(n):
    """[32, n] Mimi codes of digital silence (SIL_CODES 'seq' tiled from frame 0 = infer_ppm.sil_seq's rule)."""
    q = sil_seq()
    return q[:, np.arange(n) % q.shape[1]]


def file_sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class Pad:
    """a row's [32, 7500] int16 array, materialised on demand (np.asarray / memmap row assignment): `a` [32, n] on frames [0, n), -1
    after; a = None -> all -1. Keeps the build's RAM at the clips' size instead of the padded 7,500-frame rows."""
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a

    def __array__(self, dtype=None, copy=None):
        out = np.full((F.DEPTH, F.ROW_FRAMES), F.PAD_CODE, np.int16)
        if self.a is not None:
            out[:, :self.a.shape[1]] = self.a
        return out if dtype is None else out.astype(dtype)


# ------------------------------------------------------------------------------------------------ words
def clip_words(wl, dur, st):
    """cache words [[start, end, text]] (clip-local s) -> Model 1 side_words input, after the clip-start clamp (module doc)."""
    out = []
    for s, e, txt in wl:
        s, e = float(s), float(e)
        st["w_cache"] += 1
        if s < 0.0:
            if s < -0.5:
                st["w_before_clip_dropped"] += 1; continue
            s = 0.0; st["w_start_clamped"] += 1
        if s >= dur:
            st["w_after_clip_dropped"] += 1; continue
        out.append({"text": txt, "start": s, "end": e})
    return out


# ------------------------------------------------------------------------------------------------ plan
def _scan_shard(name):
    import pyarrow.parquet as pq
    d = f"{CACHE}/{name}"
    t = pq.read_table(f"{d}/items.parquet", columns=ITEM_COLS).to_pydict()
    off = np.load(f"{d}/codes_off.npy")
    c = np.load(f"{d}/codes.npy", mmap_mode="r"); p = np.load(f"{d}/pose.npy", mmap_mode="r")
    done = json.load(open(f"{d}/DONE.json"))
    aid = open(f"{d}/audio_id.txt").read().split("\n")
    R = len(t["key"])
    n = np.asarray(t["n"], np.int64)
    assert c.shape == p.shape == (int(off[-1]), F.DEPTH) and c.dtype == p.dtype == np.int16, (name, c.shape, p.shape)
    assert len(off) == R + 1 and off[0] == 0 and (np.diff(off) == n).all() and aid == t["key"], name
    assert done["items"] == R and done["frames"] == int(off[-1]), (name, done)
    st = collections.Counter()
    items = []
    for i in range(R):
        assert t["is_active_speaker"][i] and t["active_speaker"][i] == t["speaker"][i] and t["speaker"][i] in SIDE_OF, (name, t["key"][i])
        assert t["key"][i] == f"{t['rid'][i]}/{t['speaker'][i]}/{t['segment_id'][i]}", (name, t["key"][i])
        assert 1 <= n[i] <= F.ROW_FRAMES, (name, t["key"][i], n[i])
        wl = json.loads(t["words"][i])
        assert wl, (name, t["key"][i], "item without words")
        clip_words(wl, n[i] / F.FPS, st)
        items.append([t["rid"][i], t["key"][i], t["speaker"][i], t["segment_id"][i], int(n[i]), float(t["start"][i]), i])
    return name, items, dict(st)


def plan(a):
    import multiprocessing as mp
    P0 = json.load(open(f"{CACHE}/plan.json"))
    names = sorted(P0["shards"])
    assert len(names) == 961 and "eval" in names and all(s == "eval" or s.startswith("train-") for s in names), len(names)
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(_scan_shard, names, chunksize=4)
    CB = cache_builder()
    st = collections.Counter()
    recs = {}
    for name, items, s in res:
        st.update(s)
        assert sorted({x[0] for x in items}) == sorted(P0["shards"][name]), (name, "items' recordings != the cache plan's shard")
        for rid, key, spk, seg, n, start, i in items:
            r = recs.setdefault(rid, {"shard": name, "items": []})
            assert r["shard"] == name, (rid, "recording in two cache shards")
            r["items"].append((start, spk, seg, n, i, key))
    # held-out rule: ours == the cache builder's is_eval == the cache's eval shard, on every recording
    for rid, r in recs.items():
        ev = is_eval(rid)
        assert ev == bool(CB.is_eval(rid)) == (r["shard"] == "eval"), (rid, ev, CB.is_eval(rid), r["shard"])
        r["split"] = "eval" if ev else "train"
    keys = [x[5] for r in recs.values() for x in r["items"]]
    assert len(keys) == len(set(keys)), "duplicate item keys"
    tot = {sp: {"items": 0, "frames": 0, "recordings": 0} for sp in ("train", "eval")}
    for rid, r in recs.items():
        r["items"].sort(key=lambda x: (x[0], x[1], x[2]))          # ordinal k: clip start, then (speaker, segment_id)
        r["frames"] = sum(x[3] for x in r["items"])
        tot[r["split"]]["items"] += len(r["items"]); tot[r["split"]]["frames"] += r["frames"]; tot[r["split"]]["recordings"] += 1
    assert tot == EXPECT, (tot, EXPECT)
    S = a.stores
    load = {"eval": [0] * S, "train": [0] * S}
    for sp in ("eval", "train"):
        for rid in sorted([x for x in recs if recs[x]["split"] == sp], key=lambda x: (-recs[x]["frames"], x)):
            k = min(range(S), key=lambda k: (load[sp][k], k)); recs[rid]["store"] = k; load[sp][k] += recs[rid]["frames"]
    per = {f"p{k}": {sp: {"rows": 0, "frames": 0, "recordings": 0} for sp in ("train", "eval")} for k in range(S)}
    for rid, r in recs.items():
        q = per[f"p{r['store']}"][r["split"]]
        q["rows"] += len(r["items"]); q["frames"] += r["frames"]; q["recordings"] += 1
    for r in recs.values():                                        # data_s3q.row_arrays needs >= 2 conversation frames (counted only)
        st[f"items_n_lt2_{r['split']}"] += sum(1 for x in r["items"] if x[3] < 2)
    os.makedirs(a.out, exist_ok=True)
    out = {"format": "aloe-v2pure-plan-v1", "source": SOURCE, "cache": CACHE, "cache_builder": CACHE_BUILDER,
           "split_rule": f"eval iff blake2b8('{EVAL_SALT}' + recording_id) little-endian % {EVAL_MOD} < {EVAL_LT}", "stores": S,
           "totals": tot, "per_store": per, "load_frames": load, "word_stats_cache_scan": dict(st), "silence_codes": SIL_CODES,
           "silence_codes_sha256": file_sha256(SIL_CODES), "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "recs": {rid: {"split": r["split"], "store": r["store"], "shard": r["shard"], "frames": r["frames"],
                          "items": [[k, x[4], x[3], x[1], x[2]] for k, x in enumerate(r["items"])]}       # [k, item in shard, n, speaker, segment]
                    for rid, r in sorted(recs.items())}}
    json.dump(out, open(f"{a.out}/PLAN.json.tmp", "w"))
    os.replace(f"{a.out}/PLAN.json.tmp", f"{a.out}/PLAN.json")
    print("[plan] totals", json.dumps(tot), flush=True)
    print("[plan] per store", json.dumps(per), flush=True)
    print("[plan] word stats", json.dumps(dict(st)), flush=True)


# ------------------------------------------------------------------------------------------------ build
def build_job(job):
    """(store, cache shard, [(rid, rec plan)], tmp) -> pickle of this shard's rows for the store: per row a compact dict."""
    import pickle

    import pyarrow.parquet as pq
    from speech_q import word_pieces
    K, name, recs, tmp = job
    out = f"{tmp}/{name}.pkl"
    if os.path.exists(out):
        return out
    d = f"{CACHE}/{name}"
    t = pq.read_table(f"{d}/items.parquet", columns=ITEM_COLS).to_pydict()
    off = np.load(f"{d}/codes_off.npy")
    codes = np.load(f"{d}/codes.npy", mmap_mode="r"); pose = np.load(f"{d}/pose.npy", mmap_mode="r")
    st = collections.Counter()
    rows = []
    for rid, R in recs:
        nk = len(R["items"])
        for k, i, n, spk, seg in R["items"]:
            assert t["rid"][i] == rid and t["speaker"][i] == spk and t["segment_id"][i] == seg and int(t["n"][i]) == n, (name, rid, k, i)
            assert int(off[i + 1] - off[i]) == n
            bside = SIDE_OF[spk]; pside = "user" if bside == "assistant" else "assistant"
            cb = np.ascontiguousarray(codes[off[i]:off[i + 1]].T).astype(np.int16)
            pb = np.ascontiguousarray(pose[off[i]:off[i + 1]].T).astype(np.int16)
            assert cb.shape == pb.shape == (F.DEPTH, n), (name, rid, k, cb.shape, pb.shape)
            if int(pb.min()) < 0 or int(pb.max()) >= F.CBS:
                raise AssertionError((name, rid, k, "pose code out of [0, 2048)", int(pb.min()), int(pb.max())))
            dur = n / F.FPS
            wl = json.loads(t["words"][i])
            ws = B.side_words(clip_words(wl, dur, st), 0.0, dur, st)
            st["w_kept"] += len(ws)
            pp = word_pieces(B.tok(), ws) if ws else []
            po = np.zeros(len(ws) + 1, np.int32)
            for jj, p in enumerate(pp):
                po[jj + 1] = po[jj] + len(p)
            key = f"{B.DATASET}/{B.PODCAST}/{rid}/c{k:05d}"
            assert SPR.rec_key(key) == f"{B.DATASET}/{B.PODCAST}/{rid}", key
            split = "eval" if is_eval(rid) else "train"
            row = {"row_key": key, "dataset": B.DATASET, "dataset_folder": SOURCE, "podcast_id": B.PODCAST, "episode_id": rid,
                   "chunk_index": k, "frame_start": F.ROW_FRAMES * k, "frame_end": F.ROW_FRAMES * k + n, "n_frames": n,
                   "assistant_speaker_dir": "speaker_1", "assistant_speaker": f"{rid}/speaker_1",
                   "user_speaker_dir": "speaker_2", "user_speaker": f"{rid}/speaker_2",
                   f"words_{bside}": ws, f"piece_ids_{bside}": np.array([x for p in pp for x in p], np.int32), f"piece_offsets_{bside}": po,
                   f"words_{pside}": [], f"piece_ids_{pside}": np.zeros(0, np.int32), f"piece_offsets_{pside}": np.zeros(1, np.int32)}
            row[f"codes_{bside}"] = np.asarray(Pad(cb)); row[f"codes_{pside}"] = np.asarray(Pad(sil_codes(n)))
            F.check_row(row, strict=True)
            del row[f"codes_{bside}"], row[f"codes_{pside}"]
            row["prov"] = {"src_key": t["key"][i], "src_cache_shard": name, "src_item": int(i), "src_codes_off": int(off[i]), "split": split,
                           "recording_id": rid, "clip_start_sec": float(t["start"][i]), "clip_end_sec": float(t["end"][i]),
                           "active_speaker": spk, "segment_id": seg, "b_side": bside, "a_present": False, "partner": "absent",
                           "partner_codes": "silence", "n_clips_rec": nk, "audio_crossbleed_status": t["audio_crossbleed_status"][i] or "",
                           "lufs_status": t["lufs_status"][i] or "", "w_cache": len(wl), "w_kept": len(ws)}
            sc = {"row_key": key}
            for side in F.SIDES:
                sc.update({f"{side}_status": "missing", f"{side}_note": "no_persona_in_release", f"{side}_system_prompt": "",
                           f"{side}_sections_json": "", f"{side}_tokens_json": "", f"{side}_flags_json": "", f"{side}_version": "",
                           f"{side}_prompt_tokens": -1, f"{side}_start_sec": float(t["start"][i]), f"{side}_end_sec": float(t["start"][i]) + dur})
                st[f"prompt_{side}_missing"] += 1
            st["rows_" + split] += 1; st["frames_" + split] += n; st[f"brows_{split}_{bside}"] += 1
            rows.append({"rid": rid, "k": k, "bside": bside, "codes_b": cb, "pose_b": pb, "row": row, "side": sc})
    with open(out + ".tmp", "wb") as f:
        pickle.dump((rows, dict(st)), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(out + ".tmp", out)
    return out


def write_pose(out, name, rows, hasher):
    """pose_{user,assistant}.npy written row by row from the clips; hasher.update(file bytes) in file order (= the reference's pose_digest
    over open(file).read(), without re-reading the file)."""
    R = len(rows)
    for side in F.SIDES:
        path = f"{out}/{name}.pose_{side}.npy"
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.int16, shape=(R, F.DEPTH, F.ROW_FRAMES))
        with open(path, "rb") as f:
            hasher.update(f.read(mm.offset))
        for i, r in enumerate(rows):
            x = np.asarray(r["_pose"][side])
            mm[i] = x
            hasher.update(x.tobytes())
        mm.flush(); del mm
        assert os.path.getsize(path) == np.lib.format.open_memmap(path, mode="r").offset + R * F.DEPTH * F.ROW_FRAMES * 2


def build(a):
    import multiprocessing as mp
    import pickle
    out = f"{a.out}/p{a.store}"
    if os.path.exists(f"{out}/BUILD_DONE") and not a.force:
        print(f"[p{a.store}] BUILD_DONE exists, skipping", flush=True)
        return
    t0 = time.time()
    P = json.load(open(f"{a.out}/PLAN.json"))
    recs = sorted((rid, r) for rid, r in P["recs"].items() if r["store"] == a.store)
    if a.limit_recs:                                                   # unit check only (node /scratch output, never /data)
        assert out.startswith("/scratch/"), out
        evr = [x for x in recs if x[1]["split"] == "eval"][:a.limit_recs - a.limit_recs // 2]
        keep = {x[0] for x in recs[:a.limit_recs // 2]} | {x[0] for x in evr}
        recs = [x for x in recs if x[0] in keep]
    by_shard = collections.defaultdict(list)
    for rid, r in recs:
        by_shard[r["shard"]].append((rid, r))
    tmp = f"{TMP_ROOT}/p{a.store}" + (f"-unit{a.limit_recs}" if a.limit_recs else "")
    os.makedirs(tmp, exist_ok=True)
    jobs = [(a.store, s, by_shard[s], tmp) for s in sorted(by_shard)]
    print(f"[p{a.store}] {len(recs)} recordings in {len(jobs)} cache shards ({sum(os.path.exists(f'{tmp}/{s}.pkl') for s in by_shard)} cached)",
          flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for j, _ in enumerate(pool.imap_unordered(build_job, jobs, chunksize=2)):
            if (j + 1) % 100 == 0 or j + 1 == len(jobs):
                print(f"[p{a.store}] {j + 1}/{len(jobs)} cache shards done ({time.time() - t0:.0f}s)", flush=True)
    sil = sil_seq()
    by_rec, st = collections.defaultdict(list), collections.Counter()
    for s in sorted(by_shard):
        rr, x = pickle.load(open(f"{tmp}/{s}.pkl", "rb"))
        st.update(x)
        for r in rr:
            by_rec[r["rid"]].append(r)
    rows, sides = [], []
    for rid, R in recs:                                                # deterministic order: (rid, k)
        got = sorted(by_rec.pop(rid), key=lambda r: r["k"])
        assert [r["k"] for r in got] == list(range(len(R["items"]))), (rid, "rows != the plan's clips")
        for r in got:
            row = r["row"]
            bside = r["bside"]; pside = "user" if bside == "assistant" else "assistant"
            row[f"codes_{bside}"] = Pad(r["codes_b"]); row[f"codes_{pside}"] = Pad(sil[:, np.arange(row["n_frames"]) % sil.shape[1]])
            row["_pose"] = {bside: Pad(r["pose_b"]), pside: Pad(None)}
            row["_b_ok"] = [int(bside == "user"), int(bside == "assistant")]            # (user, assistant)
            rows.append(row); sides.append(r["side"])
    assert not by_rec, "rows of recordings outside this store"
    R = len(rows)
    exp = P["per_store"][f"p{a.store}"]
    assert a.limit_recs or (R == exp["train"]["rows"] + exp["eval"]["rows"] and st["frames_train"] == exp["train"]["frames"] and
                            st["frames_eval"] == exp["eval"]["frames"]), (R, dict(st), exp)
    print(f"[p{a.store}] {R} rows assembled ({time.time() - t0:.0f}s); writing (stats {dict(st)})", flush=True)
    extra = {"split_rows": {"eval": st["rows_eval"], "train": st["rows_train"]}, "stats": dict(st), "source": SOURCE,
             "converter": os.path.abspath(__file__), "row_writer": B.__file__, "partner": "absent (silence Mimi codes, no words, pose -1)",
             "silence_codes": SIL_CODES, "silence_codes_sha256": file_sha256(SIL_CODES)}
    meta = B.write_node(out, f"p{a.store}", rows, sides, extra)
    name = meta["shards"][0]
    print(f"[p{a.store}] node written ({time.time() - t0:.0f}s)", flush=True)
    b_ok = np.array([r["_b_ok"] for r in rows], np.uint8).reshape(R, 2)
    split = np.array([1 if r["prov"]["split"] == "eval" else 0 for r in rows], np.int8)
    assert (b_ok.sum(1) == 1).all()
    pdg = hashlib.blake2b(digest_size=16)
    write_pose(out, name, rows, pdg)
    np.savez(f"{out}/pose_index.npz", b_ok=b_ok, split=split)
    pdg.update(b_ok.tobytes()); pdg.update(split.tobytes())
    keys = [r["row_key"] for r in rows]
    json.dump({"format": "aloe-pose-addon-v1", "store": f"p{a.store}", "rows": R, "tokenizer": TOKENIZER_POSE, "normalization": NORM,
               "recipe": RECIPE, "pose_digest": pdg.hexdigest(),
               "node_digest": F.shard_digest(keys, np.array([r["n_frames"] for r in rows], np.int32)),
               "b_ok_user": int(b_ok[:, 0].sum()), "b_ok_assistant": int(b_ok[:, 1].sum()),
               "b_items_train": int(b_ok[split == 0].sum()), "b_items_eval": int(b_ok[split == 1].sum()),
               "partner_pose": "-1 (absent)", "source": SOURCE, "stats": dict(st), "built": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{out}/POSE.json", "w"), indent=1)
    open(f"{out}/BUILD_DONE", "w").write(str(R))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[p{a.store}] wrote {out}: {R} rows, train {st['rows_train']} / eval {st['rows_eval']} ({time.time() - t0:.0f}s)", flush=True)


def stores(a):
    P = json.load(open(f"{a.out}/PLAN.json"))
    out = {}
    for k in range(P["stores"]):
        root = f"{a.out}/p{k}"
        if a.limit_recs and not os.path.exists(f"{root}/BUILD_DONE"):    # unit check: only the stores built
            continue
        assert os.path.exists(f"{root}/BUILD_DONE"), root
        node = F.load_node(root)
        pj = json.load(open(f"{root}/POSE.json"))
        assert pj["node_digest"] == node["meta"]["ids_blake2b"][0] and pj["rows"] == len(node["length"]), root
        out[f"p{k}"] = {"node_digest": node["digest"], "rows": int(len(node["length"])), "pose_digest": pj["pose_digest"]}
    json.dump(out, open(f"{a.out}/STORES.json", "w"), indent=1)
    print("[stores]", json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "build", "stores"))
    ap.add_argument("--out", required=True); ap.add_argument("--stores", type=int, default=8)
    ap.add_argument("--store", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-recs", type=int, default=0, help="unit check only: build the first N recordings of the store (out on /scratch)")
    a = ap.parse_args()
    {"plan": plan, "build": build, "stores": stores}[a.cmd](a)


if __name__ == "__main__":
    main()
