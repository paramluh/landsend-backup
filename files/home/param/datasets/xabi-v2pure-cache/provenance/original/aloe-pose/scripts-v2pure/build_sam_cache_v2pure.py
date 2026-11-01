"""v2pure: build_sam_cache_active.py UNCHANGED except RAW = raw-v2pure (extracted only from releases/v2) and OUT = sam-cache-active-v2pure.
ORIGINAL DOC: ACTIVE-SPEAKER-ONLY copy of build_sam_cache.py (OUT=sam-cache-active; items with is_active_speaker False skipped first).
Aloe clean selection -> SAM 8B pose-model item caches (the EXACT format of SAM's code/train/pose_prep.py, consumed unchanged by
pose_data.PoseItems / MultiPoseItems / PosePackedDataset in train_pose.py):

  <cache>/codes.npy int16 [sum F, 32] (Mimi, frame-major), pose.npy int16 [sum F, 32] (pose tokens), codes_off.npy int64 [n+1],
  text.npy int32 [sum K] (SAM tokenizer ids, add_special_tokens=False), text_off.npy int64 [n+1], audio_id.txt (one key per line)
  + sidecar items.parquet (ignored by SAM's loader): per item key, recording, speaker, window, words with times, frame0 state for decoding.

Item = one clean Aloe speaker-segment (one speaker's window). SAM pose_prep rule, verbatim: skip iff Mimi shape != pose shape, depth != 32,
0 frames, or no non-blank words; text = " ".join(stripped non-blank words). Aloe-specific: the window's word list includes boundary-crossing
words in BOTH adjacent windows, so a word is taken into the window holding its midpoint (each word once). Items whose packed cost
(2 + K + F + 3 + SEP) exceeds the 7,500-position row are dropped here (SAM's packer would discard them at every plan; counted).

Shards: train recordings -> N_SHARDS cache dirs train-XXXX by LPT on item cost (whole recordings per shard, frames balanced);
eval recordings (blake2b8('aloe-pose-v1|' + recording_id) % 10000 < 100, same split as before) -> one cache dir 'eval'.
A node trains on POSE_DATA=<comma list of its shard dirs> (aloe_sam.shards_for_node).

usage: build_sam_cache.py plan | shards --workers W [--part K --nparts N] | finalize
"""
import argparse
import hashlib
import json
import os
import time
import traceback
from collections import Counter

import numpy as np

ROOT = "/data/home/amartyavarma/aloe-pose"
RAW = f"{ROOT}/raw-v2pure/recordings"
OUT = f"{ROOT}/sam-cache-active-v2pure"   # ACTIVE-SPEAKER ONLY variant of sam-cache (only change: listening items dropped)
TOK = "/data/home/amartyavarma/qwen-1ch/infer/sam_pose/stage1_meta/tokenizer/tokenizer.json"   # = SAM's ~/sam/tokenizer/tokenizer.json (md5 6423133b)
DEPTH, ROW_LEN, CODE_DELAY = 32, 7500, 3
EVAL_SALT, EVAL_MOD, EVAL_LT = "aloe-pose-v1|", 10000, 100
N_SHARDS = 960
JOINTS = list(range(13)) + list(range(91, 133))
FPS = 12.5
MIN_FREE_GB = 300
_TK = [None]


def tok():
    if _TK[0] is None:
        from tokenizers import Tokenizer
        _TK[0] = Tokenizer.from_file(TOK)
    return _TK[0]


def is_eval(rid):
    return int.from_bytes(hashlib.blake2b((EVAL_SALT + rid).encode(), digest_size=8).digest(), "little") % EVAL_MOD < EVAL_LT


def seg_words(words, n):
    """words of this window: midpoint inside [0, n / 12.5) (window clock); stripped, non-blank, in time order."""
    dur = n / FPS
    ws = []
    for w in words or []:
        t = str(w["word"] or "").strip()
        if not t:
            continue
        mid = 0.5 * (float(w["local_start"]) + float(w["local_end"]))
        if 0.0 <= mid < dur:
            ws.append((float(w["local_start"]), float(w["local_end"]), t))
    ws.sort(key=lambda x: (x[0], x[1]))
    return ws


def rec_items(rid, with_arrays):
    """one recording -> its items (dicts) + stats."""
    import pyarrow.parquet as pq
    cols = ["recording_id", "segment_id", "speaker", "active_speaker", "is_active_speaker", "start", "end", "start_frame", "n_codes", "words",
            "audio_crossbleed_status", "lufs_status"]
    if with_arrays:
        cols += ["mimi_tokens", "pose_tokens", "frame0_pose", "global_state0", "global_state_const", "orig_h", "orig_w"]
    t = pq.read_table(f"{RAW}/{rid}.parquet", columns=cols).to_pylist()
    st, items = Counter(), []
    for r in sorted(t, key=lambda r: (r["speaker"], r["segment_id"])):
        n = int(r["n_codes"])
        st["segments"] += 1
        if not r["is_active_speaker"]:                      # ACTIVE-ONLY: drop windows where the other person holds the floor
            st["skip_listening"] += 1; continue
        ws = seg_words(r["words"], n)
        if n == 0:
            st["skip_zero_frames"] += 1; continue
        if not ws:
            st["skip_no_words"] += 1; st["skip_no_words_codes"] += n
            st["skip_no_words_listening" if not r["is_active_speaker"] else "skip_no_words_active"] += 1
            continue
        text = " ".join(w[2] for w in ws)
        ids = np.asarray(tok().encode(text, add_special_tokens=False).ids, np.int32)
        K = len(ids)
        cost = 2 + K + n + CODE_DELAY + 1
        if cost > ROW_LEN:
            st["skip_too_long"] += 1; st["skip_too_long_codes"] += n; continue
        it = {"key": f"{rid}/{r['speaker']}/{r['segment_id']}", "rid": rid, "speaker": r["speaker"], "segment_id": r["segment_id"], "n": n, "K": K,
              "cost": cost, "is_active_speaker": bool(r["is_active_speaker"]), "active_speaker": r["active_speaker"], "start": r["start"],
              "end": r["end"], "start_frame_25fps": int(r["start_frame"]), "audio_crossbleed_status": r["audio_crossbleed_status"],
              "lufs_status": r["lufs_status"], "text": text, "words": json.dumps([[round(a, 3), round(b, 3), w] for a, b, w in ws])}
        if with_arrays:
            a = np.asarray(r["mimi_tokens"], np.int16); q = np.asarray(r["pose_tokens"], np.int16)
            if a.shape != q.shape or a.shape[0] != DEPTH or a.shape[1] != n:
                st["skip_shape"] += 1; continue
            f0 = np.asarray(r["frame0_pose"], np.float32)
            j55 = f0[JOINTS] / np.array([r["orig_h"], r["orig_w"], 1.0], np.float32)
            root, width = (j55[5, :2] + j55[6, :2]) / 2, float(abs(j55[5, 1] - j55[6, 1]))
            ok = bool(width > 0) and bool(r["global_state_const"]) and np.allclose(r["global_state0"], [*root, width], atol=1e-4)
            st["frame0_bad"] += int(not ok)
            it.update(mimi=np.ascontiguousarray(a.T), pose=np.ascontiguousarray(q.T), ids=ids, frame0_root=[float(root[0]), float(root[1])],
                      frame0_width=width, frame0_joints55=j55.reshape(-1).astype(float).tolist(), frame0_ok=ok, orig_h=float(r["orig_h"]),
                      orig_w=float(r["orig_w"]), global_state0=[float(x) for x in r["global_state0"]])
        st["items"] += 1; st["item_codes"] += n; st["item_text"] += K
        st["items_listening" if not r["is_active_speaker"] else "items_active"] += 1
        items.append(it)
    return items, st


def _plan_one(rid):
    items, st = rec_items(rid, False)
    return rid, [(i["key"], i["n"], i["K"], i["cost"]) for i in items], dict(st)


def run_plan(a):
    import multiprocessing as mp
    rids = sorted(f[:-8] for f in os.listdir(RAW) if f.endswith(".parquet"))
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(_plan_one, rids, chunksize=16)
    st = Counter(); per = {}
    for rid, its, s in res:
        st.update(s); per[rid] = its
    ev = sorted(r for r in rids if is_eval(r)); tr = [r for r in rids if not is_eval(r)]
    load = [0] * N_SHARDS
    import heapq
    h = [(0, k) for k in range(N_SHARDS)]
    shards = {f"train-{k:04d}": [] for k in range(N_SHARDS)}
    for rid in sorted(tr, key=lambda r: (-sum(x[3] for x in per[r]), r)):        # LPT: biggest recording -> least-loaded shard
        c, k = heapq.heappop(h)
        shards[f"train-{k:04d}"].append(rid); c += sum(x[3] for x in per[rid]); load[k] = c
        heapq.heappush(h, (c, k))
    shards["eval"] = ev
    for s in shards:
        shards[s].sort()
    st["recordings_train"], st["recordings_eval"] = len(tr), len(ev)
    st["shard_cost_min"], st["shard_cost_max"] = int(min(load)), int(max(load))
    os.makedirs(OUT, exist_ok=True)
    json.dump({"stats": dict(st), "shards": shards, "items": {r: per[r] for r in rids}, "built": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{OUT}/plan.json.tmp", "w"))
    os.replace(f"{OUT}/plan.json.tmp", f"{OUT}/plan.json")
    print(json.dumps(dict(st), indent=1), flush=True)


def build_shard(name, rids, expect):
    import pyarrow as pa
    import pyarrow.parquet as pq
    while True:
        sv = os.statvfs(ROOT)
        if sv.f_bavail * sv.f_frsize >= MIN_FREE_GB * 2**30:
            break
        time.sleep(60)
    t0 = time.time(); st = Counter(); items = []
    for rid in rids:
        its, s = rec_items(rid, True); st.update(s); items += its
        assert [(i["key"], i["n"], i["K"], i["cost"]) for i in its] == [tuple(x) for x in expect[rid]], (name, rid, "plan mismatch")
    d = f"{OUT}/{name}"
    tmp = d + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    frames = np.asarray([i["n"] for i in items], np.int64); ntext = np.asarray([i["K"] for i in items], np.int64)
    np.save(f"{tmp}/codes.npy", np.concatenate([i["mimi"] for i in items]).astype(np.int16))
    np.save(f"{tmp}/pose.npy", np.concatenate([i["pose"] for i in items]).astype(np.int16))
    np.save(f"{tmp}/text.npy", np.concatenate([i["ids"] for i in items]).astype(np.int32))
    np.save(f"{tmp}/codes_off.npy", np.r_[0, np.cumsum(frames)])
    np.save(f"{tmp}/text_off.npy", np.r_[0, np.cumsum(ntext)])
    with open(f"{tmp}/audio_id.txt", "w") as f:
        f.write("\n".join(i["key"] for i in items))
    side = [{k: v for k, v in i.items() if k not in ("mimi", "pose", "ids")} for i in items]
    pq.write_table(pa.Table.from_pylist(side), f"{tmp}/items.parquet", compression="zstd")
    done = {"shard": name, "items": len(items), "frames": int(frames.sum()), "text": int(ntext.sum()), "recordings": len(rids),
            "stats": dict(st), "seconds": round(time.time() - t0, 1)}
    json.dump(done, open(f"{tmp}/DONE.json", "w"))
    if os.path.exists(d):
        import shutil
        shutil.rmtree(d)
    os.replace(tmp, d)
    return name, done


def _safe(job):
    try:
        return build_shard(*job)
    except Exception:
        traceback.print_exc()
        return job[0], None


def run_shards(a):
    p = json.load(open(f"{OUT}/plan.json"))
    names = sorted(p["shards"])[a.part::a.nparts]
    todo = [(n, p["shards"][n], {r: p["items"][r] for r in p["shards"][n]}) for n in names if not os.path.exists(f"{OUT}/{n}/DONE.json")]
    print(f"[sam] part {a.part}/{a.nparts}: {len(names)} shards, {len(todo)} to build", flush=True)
    import multiprocessing as mp
    t0 = time.time(); nf = 0
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=8) as pool:
        for k, (name, d) in enumerate(pool.imap_unordered(_safe, todo)):
            if d is None:
                nf += 1; print(f"[sam] FAILED {name}", flush=True); continue
            if k % 50 == 0 or k == len(todo) - 1:
                print(f"[sam] {name} items {d['items']} {d['seconds']}s | {k + 1}/{len(todo)} fail {nf} {time.time() - t0:.0f}s", flush=True)
    print(f"[sam] DONE part {a.part} fail {nf}", flush=True)


def run_finalize(a):
    p = json.load(open(f"{OUT}/plan.json"))
    man, agg = {}, {"train": Counter(), "eval": Counter()}
    for n in sorted(p["shards"]):
        d = json.load(open(f"{OUT}/{n}/DONE.json"))
        sp = "eval" if n == "eval" else "train"
        files = {f: os.path.getsize(f"{OUT}/{n}/{f}") for f in sorted(os.listdir(f"{OUT}/{n}"))}
        man[n] = {k: d[k] for k in ("items", "frames", "text", "recordings")} | {"files": files}
        agg[sp].update({k: d[k] for k in ("frames", "text", "recordings")}); agg[sp].update(d["stats"]); agg[sp]["shards"] += 1
    summ = {sp: dict(agg[sp]) | {"speech_hours": round(agg[sp]["frames"] / FPS / 3600, 2),
                                 "positions": int(agg[sp]["frames"] + agg[sp]["text"] + agg[sp]["items"] * (2 + CODE_DELAY + 1))} for sp in agg}
    json.dump({"format": "SAM pose_prep item cache (codes/pose/text/offsets/audio_id) + items.parquet sidecar", "row_len": ROW_LEN,
               "tokenizer": TOK, "eval_rule": f"blake2b8('{EVAL_SALT}' + recording_id) % {EVAL_MOD} < {EVAL_LT}", "summary": summ,
               "plan_stats": p["stats"], "shards": man}, open(f"{OUT}/manifest.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "shards", "finalize"])
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    {"plan": run_plan, "shards": run_shards, "finalize": run_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
