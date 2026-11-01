"""FRESH POSE for the 8,1,1 resume (10-21 PT, user: "completely fresh pose data, the HF bucket new-pose-llm-data"): new-pose-llm-data parquet
(inline pose / audio structs; FORMAT.md in data/new-pose-llm-data-dl) -> PPM pose rows ('s3-pose-rows-v1' rows, EXACTLY what
pose_s3q.PoseRows.__getitem__ returns, built with qwen-1ch-data/pose-s3-samdata/scripts/build_pose_rows_sam.py's own functions: same SAM
filter, same word placement / NEW_WORD lane (stage-2 speech_q, Qwen3-32B tokenizer), same [SLOT 1 | clip | GAP 4] framing in 7,500-frame rows,
same per-video units + first-fit-decreasing packing, same frame0 fields). Pose tokenizer InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f
(= PPM's rows; checked per segment as a counted flag, as the builder), Mimi kyutai/mimi 32 x 2048 @ 12.5 Hz.

ROWS ARE BUILT ON THE FLY from the node-local parquet (no second copy: the hosts have 31-49 GB free and every 8,1,1 save needs ~52 GB). The
ROW PLAN is fixed once (PLAN.json + PLAN.npz per host): every row = an ordered list of segments, so a row's content is a pure function of
(plan, parquet) -> resume-exact.

FILTERS (coordinator 10-21): _canary never read (only jobs/); a release is used iff its parquet files' size + sha256 equal the pins of its
complete.json (if present) else its release.json (no pin file -> rejected); its state must be completed / completed_with_exceptions / partial;
duplicate segments across releases / jobs / hosts: first occurrence in the GLOBAL order (hosts sorted, jobs sorted, releases: sealed first, then
completed < completed_with_exceptions < partial, then name; files / rows in order); DROPPED: every segment whose video_id is in PPM's
TikTok-pose-data v2 + v3 pose set (tools/ppm_tiktok_vids.py: the clip_id prefix of every source row = the clip set PPM's samdata stores were
built from), every pose-eval video (blake2b8('pose-s3-v1|' + video_id) % 10000 < 10, PPM's eval rule), and SAM's pose_prep filter (decided
at scan time with the real decode, asserted again when a row is built).

usage (python -I, on the host that holds the data):  fresh_pose.py scan | plan | all [--workers N] | check [--rows i,j,k]

10-22 APPENDED PLANS (user: "more distinct pose data soon, same format; tack it onto the end of the data schedule"; fresh/append/ingest_more.sh):
the same scan / plan / check over another node-local copy (FRESH_SRC, FRESH_OUT, FRESH_SCAN) with FRESH_TAG=<tag> (e.g. npl2: row keys
'<tag>/<host>/..', dataset '<TAG>', its own pack seed) and FRESH_PRIOR_SCANS=<scan dir>[,..] (the earlier plans' scans, e.g. fresh/scan):
a segment is ALSO dropped if an earlier plan scanned its segment_id or its clip_id, and (FRESH_PRIOR_VIDEO, default overlap) if its video is
in an earlier plan and its parent-clip source span (clip_id '<video>_MM-SS.ss_to_MM-SS.ss') overlaps any earlier clip span of that video
(unparseable -> dropped); 'video' drops every earlier video, 'off' = segment + clip ids only. Every other rule (release pins, state, segment
dedup, PPM TikTok v2+v3 videos, eval videos, SAM filter) is plan 1's. The trainer appends such plans with PPM_FRESH_POSE_ROOT2 (rows tagged
k << 40: append_stream / FreshPoseMultiDS). Unset FRESH_TAG / FRESH_PRIOR_SCANS = plan 1's behaviour exactly.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
import zlib
from collections import Counter, OrderedDict, defaultdict

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch-data/pose-s3-samdata/scripts")
import build_pose_rows_sam as B  # noqa: E402  (units_of / pack / build_row / is_eval / tok / constants; imported, not edited)

SP = B.SP
SRC_ROOT = os.environ.get("FRESH_SRC", "/scratch/amartyavarma/new-pose-llm-data/new-pose-llm-data/jobs")
OUT = os.environ.get("FRESH_OUT", "/scratch/amartyavarma/ppm-fresh-pose")
FRESH = "/data/home/amartyavarma/ppm/code-811/fresh"
SCAN = os.environ.get("FRESH_SCAN", f"{FRESH}/scan")
HOSTS = os.environ["FRESH_HOSTS"].split(",") if os.environ.get("FRESH_HOSTS") else \
    ["slinky-0", "slinky-1", "slinky-2", "slinky-3", "slinky-5", "slinky-6", "slinky-7", "slinky-8", "slinky-9", "slinky-11", "slinky-12",
     "slinky-13", "slinky-14", "slinky-16", "slinky-17", "slinky-19"]
HOST = os.environ.get("FRESH_HOST", os.uname().nodename.split(".")[0])
VERSION = "ppm-fresh-pose-npl-v1"
STATE_RANK = {"completed": 0, "completed_with_exceptions": 1, "partial": 2}
SEG_PER_SHARD = int(os.environ.get("FRESH_SEG_PER_SHARD", "2500"))
PACK_SEED = 20261022
COLS = ["segment_id", "clip_id", "video_id", "bucket", "pose", "audio"]
TAG = os.environ.get("FRESH_TAG", "")                       # '' = plan 1 (10-21); else an appended plan (doc above)
PRIOR_SCANS = [x for x in os.environ.get("FRESH_PRIOR_SCANS", "").split(",") if x]
PRIOR_VIDEO = os.environ.get("FRESH_PRIOR_VIDEO", "overlap")
assert PRIOR_VIDEO in ("overlap", "video", "off"), PRIOR_VIDEO
assert not TAG or (re.fullmatch(r"[a-z][a-z0-9-]*", TAG) and TAG != "npl" and PRIOR_SCANS), \
    ("an appended plan needs FRESH_TAG=<lowercase tag, not npl> and FRESH_PRIOR_SCANS=<earlier plans' scan dirs>", TAG, PRIOR_SCANS)
assert not PRIOR_SCANS or TAG, "FRESH_PRIOR_SCANS is for an appended plan: set FRESH_TAG"
PLAN_SHIFT = 40                                              # trainer stream ids: plan k >= 1 rows = (k << 40) | row (plan 1 rows plain)
PRIOR_REASONS = {1: "segment", 2: "clip", 3: "video_overlap", 4: "video", 5: "video_unparsed"}
SPAN_EPS = 0.04                                              # s: spans that only touch (consecutive clips) do not overlap
CLIP_RE = re.compile(r"^(\d+)_(\d+)-(\d+(?:\.\d+)?)_to_(\d+)-(\d+(?:\.\d+)?)$")


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{HOST}]", *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ one source segment -> clip
def decode_new(r, st):
    """new-pose-llm-data row (inline structs) -> clip dict (build_pose_rows_sam.decode_clip's fields), or None iff SAM's pose_prep filter
    drops it (mimi shape != pose shape, depth != 32, 0 frames, no non-blank word_texts) or it cannot be represented (code range, too long).
    Counted-only checks as in the builder (flag_*)."""
    P, A = r["pose"], r["audio"]
    if P is None or A is None or A.get("mimi_codes") is None:
        st["drop_missing_artifact"] += 1; return None
    mc = A["mimi_codes"]
    lens = [len(x) for x in mc]
    if not lens or any(x != lens[0] for x in lens):
        st["sam_drop_shape"] += 1; st["drop_ragged_mimi"] += 1; return None
    T, nc = lens[0], len(lens)
    pl = [P.get(f"tokens_cb{i}") for i in range(B.DEPTH)]
    if any(x is None for x in pl) or any(len(x) != len(pl[0]) for x in pl):
        st["sam_drop_shape"] += 1; st["drop_ragged_pose"] += 1; return None
    pshape = (len(pl), len(pl[0]))
    wt_raw = A.get("word_texts") or []
    sam_words = [w.strip() for w in wt_raw if w is not None and w.strip()]
    if (nc, T) != pshape or nc != B.DEPTH or T == 0 or not sam_words:
        if (nc, T) != pshape or nc != B.DEPTH:
            st["sam_drop_shape"] += 1
        elif T == 0:
            st["sam_drop_zero_frames"] += 1
        else:
            st["sam_drop_no_words"] += 1; st["frames_sam_drop_no_words"] += T
        return None
    codes = np.asarray(mc, np.int64); pose = np.asarray(pl, np.int64)
    if A.get("num_frames") is not None and int(A["num_frames"]) != T:
        st["flag_num_frames_ne_codes"] += 1
    if bool(P.get("skipped")):
        st["flag_pose_skipped"] += 1
    if P.get("n_code_steps") is not None and int(P["n_code_steps"]) != T:
        st["flag_pose_n_code_steps_ne"] += 1
    if P.get("token_time_anchor") != "end_of_two_frame_group" or abs(float(P.get("token_stride_s") or 0) - 0.08) > 1e-9 or \
       P.get("normalization_recipe") != "frame0-root-frame0-horizontal-shoulder-width-v1" or \
       P.get("checkpoint_revision") != "1d6ff4f0de89408cb054da618e309d4334596252":
        st["flag_pose_contract"] += 1
    if int(A.get("audio_sr") or 24000) != 24000:
        st["flag_audio_sr"] += 1
    if int(codes.min()) < 0 or int(codes.max()) >= B.CBS or int(pose.min()) < 0 or int(pose.max()) >= B.CBS:
        st["drop_code_range"] += 1; return None
    if T + B.SLOT + B.GAP > B.ROW_FRAMES:
        st["drop_too_long"] += 1; return None
    frame0_ok, H, W, root, width, j55 = False, 0.0, 0.0, np.zeros(2, np.float32), 0.0, np.zeros((55, 3), np.float32)
    try:
        gs = np.asarray(P["global_state"], np.float32).reshape(-1, 3)
        f0 = np.asarray(P["frame0_pose"], np.float32).reshape(-1, 3)
        H, W = float(P["orig_h"]), float(P["orig_w"])
        if len(gs) and np.isfinite(gs).all() and (gs == gs[0]).all() and f0.shape == (133, 3) and np.isfinite(f0).all() and H > 0 and W > 0:
            j55 = f0[B.JOINTS] / np.array([H, W, 1.0], np.float32)
            root, width = (j55[5, :2] + j55[6, :2]) / 2, abs(j55[5, 1] - j55[6, 1])
            frame0_ok = bool(width > 0) and np.allclose(gs[0], [*root, width], atol=1e-4)
    except Exception:  # noqa: BLE001
        frame0_ok = False
    if not frame0_ok:
        st["flag_frame0_bad"] += 1
    if T < 38:
        st["kept_short_lt3s"] += 1; st["frames_kept_short_lt3s"] += T
    lang = A.get("language")
    st["lang_" + str(lang or "none")] += 1
    vid = str(r["video_id"])
    if not str(r["clip_id"]).startswith(vid):
        st["flag_vid_not_clip_prefix"] += 1
    ws_, we_ = A.get("word_start_seconds") or [], A.get("word_end_seconds") or []
    assert len(wt_raw) == len(ws_) == len(we_), ("word arrays", r["segment_id"])
    dur = T / B.FPS
    words = []
    for txt, s0_, e0_ in zip(wt_raw, ws_, we_):
        txt = str(txt or "").strip()
        if not txt:
            st["w_empty"] += 1; continue
        if s0_ is None or e0_ is None:
            st["w_nonfinite"] += 1; continue
        s0_, e0_ = float(s0_), float(e0_)
        if not (math.isfinite(s0_) and math.isfinite(e0_)):
            st["w_nonfinite"] += 1; continue
        if s0_ < 0.0:
            if s0_ < -0.5:
                st["w_before_clip"] += 1; continue
            s0_ = 0.0; st["w_clamped_start"] += 1
        if s0_ >= dur:
            st["w_after_clip"] += 1; continue
        words.append({"text": txt, "start": s0_, "end": min(max(e0_, s0_), dur)})
    if any(y["start"] < x["start"] for x, y in zip(words, words[1:])):
        st["clips_resorted_words"] += 1
        words.sort(key=lambda w: (w["start"], w["end"]))
    if not words:
        st["kept_clip_no_placeable_words"] += 1
    return {"codes": codes.astype(np.int16), "pose": pose.astype(np.int16), "n": T, "words": words, "video_id": vid, "clip_id": str(r["clip_id"]),
            "segment_id": str(r["segment_id"]), "parent_clip_id": str(A.get("parent_clip_id") or r["clip_id"]),
            "segment_index": int(A.get("segment_index") or 0), "partition": str(r["bucket"]), "language": lang,
            "src_start_s": float(A["source_start_s"]) if A.get("source_start_s") is not None else -1.0,
            "src_end_s": float(A["source_end_s"]) if A.get("source_end_s") is not None else -1.0, "orig_h": H, "orig_w": W,
            "frame0_root": [float(root[0]), float(root[1])], "frame0_width": float(width), "frame0_joints55": j55.reshape(-1).astype(float).tolist(),
            "frame0_ok": frame0_ok, "n_sam_words": len(sam_words)}


# ------------------------------------------------------------------------------------------------ scan (per host)
def scan_release(args):
    job, rel = args
    import pyarrow.parquet as pq
    d = f"{SRC_ROOT}/{job}/{rel}"
    fs = sorted(os.listdir(d))
    st = Counter()
    sealed = "complete.json" in fs
    state, pins = None, None
    try:
        if sealed:
            c = json.load(open(f"{d}/complete.json")); pins = c.get("files"); state = c.get("state")
        if "release.json" in fs:
            r = json.load(open(f"{d}/release.json")); state = state or r.get("state")
            if pins is None:
                pins = r.get("files")
    except Exception as e:  # noqa: BLE001
        return job, rel, None, {"rel_bad_json": 1}, repr(e)[:200]
    parts = [f for f in fs if f.startswith("part-") and f.endswith(".parquet")]
    if pins is None:
        st["rel_no_pins"] += 1; st["rel_no_pins_parts"] += len(parts)
        return job, rel, None, dict(st), "no pins"
    if state not in STATE_RANK:
        st["rel_bad_state"] += 1
        return job, rel, None, dict(st), f"state {state}"
    pinned = sorted(k for k in pins if k.endswith(".parquet"))
    if sorted(parts) != pinned:
        st["rel_parts_mismatch"] += 1
        return job, rel, None, dict(st), f"parts {parts} vs pins {pinned}"
    out = []
    for p in parts:
        fp = f"{d}/{p}"
        if os.path.getsize(fp) != int(pins[p]["bytes"]) or sha256(fp) != pins[p]["sha256"]:
            st["rel_hash_mismatch"] += 1
            return job, rel, None, dict(st), f"hash {p}"
        rows = pq.read_table(fp, columns=COLS).to_pylist()
        if "rows" in pins[p] and int(pins[p]["rows"]) != len(rows):
            st["rel_rows_mismatch"] += 1
            return job, rel, None, dict(st), f"rows {p}"
        for i, r in enumerate(rows):
            c = decode_new(r, st)
            A = r["audio"] or {}
            out.append((p, i, str(r["segment_id"]), str(r["video_id"]), str(r["clip_id"]), str(r["bucket"]),
                        str(A.get("parent_clip_id") or r["clip_id"]), int(A.get("segment_index") or 0),
                        int(c["n"]) if c is not None else int(A.get("num_frames") or 0), c is not None))
    st["rel_ok"] += 1; st["rel_ok_sealed"] += int(sealed); st[f"rel_ok_{state}"] += 1; st["segments"] += len(out)
    return job, rel, {"sealed": sealed, "state": state, "rows": out}, dict(st), ""


def cmd_scan(a):
    import multiprocessing as mp
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(SCAN, exist_ok=True)
    if os.path.exists(f"{SCAN}/{HOST}.json"):
        log("scan exists"); return
    jobs = sorted(os.listdir(SRC_ROOT))[:int(os.environ.get("FRESH_LIMIT_JOBS", "0")) or None]
    todo = [(j, r) for j in jobs for r in sorted(os.listdir(f"{SRC_ROOT}/{j}"))]
    log(f"scan: {len(jobs)} jobs, {len(todo)} releases, {a.workers} workers")
    res, agg, bad = {}, Counter(), []
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (job, rel, r, st, why) in enumerate(pool.imap_unordered(scan_release, todo, chunksize=8)):
            agg.update(st); res[(job, rel)] = r
            if why and len(bad) < 50:
                bad.append((job, rel, why))
            if k % 4000 == 0:
                log(f"scan {k}/{len(todo)} {time.time() - t0:.0f}s")
    cols = defaultdict(list)
    for job in jobs:                                          # this host's order: jobs sorted, releases by priority, files / rows in order
        rels = [(r, res[(job, r)]) for r in sorted(os.listdir(f"{SRC_ROOT}/{job}")) if res.get((job, r)) is not None]
        rels.sort(key=lambda x: (0 if x[1]["sealed"] else 1, STATE_RANK[x[1]["state"]], x[0]))
        for pr, (rel, r) in enumerate(rels):
            for (p, i, s, v, c, b, pc, si, n, ok) in r["rows"]:
                for k_, x in (("file", f"{job}/{rel}/{p}"), ("row", i), ("rel_prio", pr), ("segment_id", s), ("video_id", v), ("clip_id", c),
                              ("bucket", b), ("parent_clip_id", pc), ("segment_index", si), ("n", n), ("sam_ok", ok), ("sealed", r["sealed"]),
                              ("state", r["state"])):
                    cols[k_].append(x)
    if not cols:                                              # (an appended plan's host that got no files: typed empty scan)
        ty = {"file": pa.string(), "row": pa.int64(), "rel_prio": pa.int64(), "segment_id": pa.string(), "video_id": pa.string(),
              "clip_id": pa.string(), "bucket": pa.string(), "parent_clip_id": pa.string(), "segment_index": pa.int64(), "n": pa.int64(),
              "sam_ok": pa.bool_(), "sealed": pa.bool_(), "state": pa.string()}
        cols = {k: pa.array([], t_) for k, t_ in ty.items()}
    pq.write_table(pa.table(dict(cols)), f"{SCAN}/{HOST}.parquet.tmp"); os.replace(f"{SCAN}/{HOST}.parquet.tmp", f"{SCAN}/{HOST}.parquet")
    info = {"host": HOST, "jobs": len(jobs), "releases": len(todo), "stats": dict(agg), "bad_examples": bad, "segments": len(cols["segment_id"]),
            "frames": int(sum(cols["n"])), "seconds": round(time.time() - t0, 1)}
    json.dump(info, open(f"{SCAN}/{HOST}.json.tmp", "w"), indent=1); os.replace(f"{SCAN}/{HOST}.json.tmp", f"{SCAN}/{HOST}.json")
    log(f"scan done: {info['segments']:,} segments {info['frames'] / 12.5 / 3600:.0f} h {info['seconds']}s; {dict(agg)}")


# ------------------------------------------------------------------------------------------------ appended plans: dedup against earlier plans
def h64s(xs):
    """strings -> uint64 blake2b-8 hashes (prior-plan membership; a false match only drops a segment)."""
    return np.fromiter((int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "little") for x in xs), np.uint64, len(xs))


def clip_span(clip_id, video_id):
    """parent-clip source span (s) from '<video>_MM-SS.ss_to_MM-SS.ss', or None (unparseable / other video / empty)."""
    m = CLIP_RE.match(clip_id)
    if not m or m.group(1) != video_id:
        return None
    a, b = int(m.group(2)) * 60 + float(m.group(3)), int(m.group(4)) * 60 + float(m.group(5))
    return (a, b) if b > a else None


def prior_index(new_vh):
    """earlier plans' scans (EVERY scanned segment, kept or not) -> {seg: sorted segment-id hashes, clip: sorted clip-id hashes, spans: video
    hash -> merged disjoint sorted [(a, b)] (None = an unparseable earlier clip: the whole video), restricted to the videos in new_vh}."""
    import pyarrow.parquet as pq
    seg, clip, vrows, files = [], [], [], []
    for d in PRIOR_SCANS:
        fs = sorted(glob.glob(f"{d}/*.parquet"))
        assert fs, ("no prior scans in", d)
        files += fs
        for f in fs:
            t = pq.read_table(f, columns=["segment_id", "clip_id", "video_id"]).to_pydict()
            seg.append(h64s(t["segment_id"])); clip.append(h64s(t["clip_id"]))
            vh = h64s(t["video_id"]); m = np.flatnonzero(np.isin(vh, new_vh))
            vrows += [(int(vh[i]), clip_span(t["clip_id"][i], t["video_id"][i])) for i in m]
    spans = {}
    for v, sp in vrows:
        spans.setdefault(v, []).append(sp)
    for v, l in spans.items():
        if any(x is None for x in l):
            spans[v] = None; continue
        l.sort(); mg = [list(l[0])]
        for a, b in l[1:]:
            if a < mg[-1][1] - SPAN_EPS:
                mg[-1][1] = max(mg[-1][1], b)
            else:
                mg.append([a, b])
        spans[v] = (np.array([x[0] for x in mg]), np.array([x[1] for x in mg]))
    return {"seg": np.unique(np.concatenate(seg)), "clip": np.unique(np.concatenate(clip)), "spans": spans, "files": files,
            "span_keys": np.fromiter(spans.keys(), np.uint64, len(spans)), "n_prior_segments": int(sum(len(x) for x in seg))}


def prior_drop(pri, t):
    """-> (uint8 reason per new scan row: 0 = keep, else PRIOR_REASONS; bool: its video is in an earlier plan, its span is new)."""
    sh, ch, vh = h64s(t["segment_id"]), h64s(t["clip_id"]), h64s(t["video_id"])
    r = np.zeros(len(sh), np.uint8)
    r[np.isin(ch, pri["clip"])] = 2
    r[np.isin(sh, pri["seg"])] = 1
    newspan = np.zeros(len(sh), bool)
    if PRIOR_VIDEO != "off":
        for i in np.flatnonzero((r == 0) & np.isin(vh, pri["span_keys"])):
            sp, q = pri["spans"][int(vh[i])], clip_span(t["clip_id"][i], t["video_id"][i])
            if PRIOR_VIDEO == "video":
                r[i] = 4
            elif sp is None or q is None:
                r[i] = 5
            else:
                a, b = sp
                j = int(np.searchsorted(a, q[1] - SPAN_EPS, "left")) - 1   # last earlier span starting before q ends (and the one before)
                if any(jj >= 0 and b[jj] > q[0] + SPAN_EPS for jj in (j, j - 1)):
                    r[i] = 3
                else:
                    newspan[i] = True
    return r, newspan


# ------------------------------------------------------------------------------------------------ plan (global dedup; every host computes the same)
def global_keep():
    """-> (dict host -> boolean keep mask over its scan rows, stats). Deterministic: identical on every host."""
    import pyarrow.parquet as pq
    ppm = np.load(f"{FRESH}/ppm_tiktok_video_ids.npy")
    seen, keep, st = set(), {}, Counter()
    pri = None
    if PRIOR_SCANS:                                           # appended plan (doc above): the earlier plans' ids / spans
        t0_ = time.time()
        nv_ = np.unique(np.concatenate([h64s(pq.read_table(f"{SCAN}/{h}.parquet", columns=["video_id"]).column(0).to_pylist()) for h in HOSTS]))
        pri = prior_index(nv_)
        st["prior_segments_scanned"] = pri["n_prior_segments"]; st["prior_scan_files"] = len(pri["files"])
        st["prior_videos_shared"] = len(pri["spans"])
        log(f"prior plans {PRIOR_SCANS} ({len(pri['files'])} scans): {pri['n_prior_segments']:,} segments, {len(pri['clip']):,} clips; "
            f"{len(pri['spans']):,} of the {len(nv_):,} new videos are in them; mode {PRIOR_VIDEO}; {time.time() - t0_:.0f}s")
    for h in HOSTS:
        t = pq.read_table(f"{SCAN}/{h}.parquet", columns=["segment_id", "video_id", "n", "sam_ok"] + (["clip_id"] if pri else [])).to_pydict()
        pdrop = None
        if pri is not None:
            pdrop, pnew = prior_drop(pri, t)
        vid = t["video_id"]
        nonnum = np.array([not v.isdigit() for v in vid], bool)
        num = np.array([int(v) if v.isdigit() else 0 for v in vid], np.uint64)
        in_ppm = np.isin(num, ppm) & ~nonnum
        ev = np.array([B.is_eval(v) for v in vid], bool)
        n = np.asarray(t["n"], np.int64)
        ok = np.asarray(t["sam_ok"], bool)
        k = np.zeros(len(vid), bool)
        for i, s in enumerate(t["segment_id"]):
            st["scanned"] += 1; st["frames_scanned"] += int(n[i])
            if s in seen:
                st["drop_dup_segment"] += 1; st["frames_drop_dup_segment"] += int(n[i]); continue
            seen.add(s)
            st["unique"] += 1; st["frames_unique"] += int(n[i])
            if pdrop is not None and pdrop[i]:                  # appended plan: already in an earlier plan (doc above)
                r_ = PRIOR_REASONS[int(pdrop[i])]
                st[f"drop_prior_{r_}"] += 1; st[f"frames_drop_prior_{r_}"] += int(n[i]); continue
            if in_ppm[i]:
                st["drop_ppm_video"] += 1; st["frames_drop_ppm_video"] += int(n[i]); continue
            if ev[i]:
                st["drop_eval_video"] += 1; st["frames_drop_eval_video"] += int(n[i]); continue
            if not ok[i]:
                st["drop_sam_filter"] += 1; st["frames_drop_sam_filter"] += int(n[i]); continue
            k[i] = True; st["kept"] += 1; st["frames_kept"] += int(n[i]); st[f"kept_{h}"] += 1; st[f"frames_kept_{h}"] += int(n[i])
            if pdrop is not None and pnew[i]:
                st["kept_prior_video_new_span"] += 1; st["frames_kept_prior_video_new_span"] += int(n[i])
        st["nonnumeric_video_ids"] += int(nonnum.sum())
        keep[h] = k
    return keep, st


def cmd_plan(a):
    import pyarrow.parquet as pq
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUT}/PLAN.json"):
        log("plan exists"); return
    while not all(os.path.exists(f"{SCAN}/{h}.json") for h in HOSTS):
        log("waiting for scans: missing " + ",".join(h for h in HOSTS if not os.path.exists(f"{SCAN}/{h}.json"))); time.sleep(10)
    t0 = time.time()
    keep, st = global_keep()
    t = pq.read_table(f"{SCAN}/{HOST}.parquet").to_pydict()
    k = np.flatnonzero(keep[HOST])
    files = sorted({t["file"][i] for i in k})
    fidx = {f: j for j, f in enumerate(files)}
    seg = {"file": np.array([fidx[t["file"][i]] for i in k], np.int32), "row": np.array([t["row"][i] for i in k], np.int32),
           "n": np.array([t["n"][i] for i in k], np.int32)}
    sid = [t["segment_id"][i] for i in k]
    clips = [{"video_id": t["video_id"][i], "parent_clip_id": t["parent_clip_id"][i], "segment_index": int(t["segment_index"][i]),
              "segment_id": t["segment_id"][i], "n": int(t["n"][i])} for i in k]
    shards, cur, last_job = [], [], None                   # consecutive kept segments in host order; a shard closes only at a job boundary
    for j in range(len(k)):
        job = t["file"][k[j]].split("/")[0]
        if len(cur) >= SEG_PER_SHARD and job != last_job:
            shards.append(cur); cur = []
        cur.append(j); last_job = job
    if cur:
        shards.append(cur)
    rows, row_shard = [], []
    for s, idx in enumerate(shards):
        rng = np.random.default_rng(np.random.SeedSequence([PACK_SEED, zlib.crc32(HOST.encode()), s] + ([zlib.crc32(TAG.encode())] if TAG else [])))
        sub = [clips[j] for j in idx]
        for cl in B.pack(B.units_of(sub), rng):
            rows.append([idx[c] for c in cl]); row_shard.append(s)
    used = sorted(j for r in rows for j in r)
    assert used == list(range(len(k))), "every kept segment in exactly one row"
    row_len = np.array([sum(int(seg["n"][j]) + B.SLOT + B.GAP for j in r) for r in rows], np.int64)
    assert (row_len <= B.ROW_FRAMES).all()
    row_seg = np.array([j for r in rows for j in r], np.int32)
    row_off = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
    h = hashlib.blake2b(digest_size=16)
    for f in files:
        h.update(f.encode()); h.update(b"\0")
    for x in (seg["file"], seg["row"], seg["n"], row_seg, row_off):
        h.update(np.ascontiguousarray(x).tobytes())
    np.savez(f"{OUT}/PLAN.npz.tmp.npz", seg_file=seg["file"], seg_row=seg["row"], seg_n=seg["n"], row_seg=row_seg, row_off=row_off,
             row_len=row_len, row_shard=np.array(row_shard, np.int32))
    os.replace(f"{OUT}/PLAN.npz.tmp.npz", f"{OUT}/PLAN.npz")
    plan = {"version": VERSION, "format": B.FORMAT, "host": HOST, "hosts": HOSTS, "src_root": SRC_ROOT, "files": files, "segment_ids": sid,
            "seg_per_shard": SEG_PER_SHARD, "n_shards": len(shards), "n_rows": len(rows), "kept_segments": int(len(k)),
            "kept_frames": int(seg["n"].sum()), "row_frames_total": int(row_len.sum()), "digest": h.hexdigest(), "global": dict(st),
            "row_frames": B.ROW_FRAMES, "slot": B.SLOT, "gap": B.GAP, "depth": B.DEPTH, "tokenizer": B.TOKENIZER,
            "pose_tokenizer": "InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0de89408cb054da618e309d4334596252",
            "builder": os.path.abspath(__file__), "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **({"tag": TAG, "key_prefix": TAG, "dataset": TAG.upper(), "prior_scans": PRIOR_SCANS, "prior_video": PRIOR_VIDEO} if TAG else {})}
    with open(f"{OUT}/PLAN.json.tmp", "w") as f:
        json.dump(plan, f); f.flush(); os.fsync(f.fileno())
    os.replace(f"{OUT}/PLAN.json.tmp", f"{OUT}/PLAN.json")
    os.makedirs(f"{SCAN}/plans", exist_ok=True)                 # audit copy (Weka)
    json.dump({kk: v for kk, v in plan.items() if kk not in ("files", "segment_ids")}, open(f"{SCAN}/plans/PLAN-{HOST}.json", "w"), indent=1)
    log(f"plan: {plan['kept_segments']:,} segments {plan['kept_frames'] / 12.5 / 3600:.0f} h -> {len(rows):,} rows ({len(rows) // 64} steps at 64 rows / "
        f"host-step) in {len(shards)} shards, fill {row_len.mean() / B.ROW_FRAMES:.3f}, {time.time() - t0:.0f}s; global {dict(st)}")


# ------------------------------------------------------------------------------------------------ rows (used by the trainer)
def load_plan(root):
    meta = json.load(open(f"{root}/PLAN.json"))
    assert meta["version"] == VERSION and meta["format"] == B.FORMAT, (meta.get("version"), meta.get("format"))
    z = dict(np.load(f"{root}/PLAN.npz"))
    h = hashlib.blake2b(digest_size=16)
    for f in meta["files"]:
        h.update(f.encode()); h.update(b"\0")
    for x in ("seg_file", "seg_row", "seg_n", "row_seg", "row_off"):
        h.update(np.ascontiguousarray(z[x]).tobytes())
    assert h.hexdigest() == meta["digest"], ("fresh pose plan digest", root)
    meta["z"] = z
    return meta


class FreshRows:
    """rows[i] -> the dict pose_s3q.PoseRows.__getitem__ returns (codes [32, 7500] int16, pose, lane int32, kind int8, row, n_frames, row_key,
    clips, split, dataset, n_words, n_pieces), built from the plan's segments with build_pose_rows_sam.build_row."""

    def __init__(self, plan, cache_files=64):
        self.p, self.z, self.cache, self.nc = plan, plan["z"], OrderedDict(), int(cache_files)

    def __len__(self):
        return int(self.p["n_rows"])

    def _table(self, fi):
        import pyarrow.parquet as pq
        if fi not in self.cache:
            self.cache[fi] = pq.read_table(f"{self.p['src_root']}/{self.p['files'][fi]}", columns=COLS)
            while len(self.cache) > self.nc:
                self.cache.popitem(last=False)
        self.cache.move_to_end(fi)
        return self.cache[fi]

    def clips_of(self, i):
        z = self.z
        segs = z["row_seg"][z["row_off"][i]:z["row_off"][i + 1]]
        st, clips = Counter(), []
        for j in segs:
            r = self._table(int(z["seg_file"][j])).slice(int(z["seg_row"][j]), 1).to_pylist()[0]
            assert r["segment_id"] == self.p["segment_ids"][j], ("fresh pose: plan segment != parquet row", r["segment_id"], self.p["segment_ids"][j])
            c = decode_new(r, st)
            assert c is not None and c["n"] == int(z["seg_n"][j]), ("fresh pose: a planned segment no longer decodes", r["segment_id"])
            c["pieces"] = SP.word_pieces(B.tok(), c["words"]) if c["words"] else []
            clips.append(c)
        return clips, st

    def __getitem__(self, i):
        i = int(i)
        clips, st = self.clips_of(i)
        codes, pose, kind, lane, words, pids, offs, crec, nf = B.build_row(clips, list(range(len(clips))), st)
        assert nf == int(self.z["row_len"][i]), ("fresh pose row length", i, nf, int(self.z["row_len"][i]))
        return {"codes": codes, "pose": pose, "lane": lane, "kind": kind, "row": i, "n_frames": int(nf),
                "row_key": f"{self.p.get('key_prefix', 'npl')}/{self.p['host']}/{int(self.z['row_shard'][i]):05d}/train/{i:06d}", "clips": crec,
                "split": "train", "dataset": self.p.get("dataset", "NPL"), "n_words": len(words), "n_pieces": len(pids)}


class FreshPoseDS:
    """the trainer's pose dataset over fresh rows (= data_ppm.PoseDS with FreshRows in place of PoseRows): item i = packed(pose_micro_ppm(row))."""

    def __init__(self, plan, rows, cfg, rank):
        self.plan, self.rows, self.cfg, self.rank, self.fr = plan, [int(r) for r in rows], dict(cfg), int(rank), None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import data_ppm as D                                    # the trainer's module (sys.modules)
        if self.fr is None:
            self.fr = FreshRows(self.plan, cache_files=48)
        return D.packed(D.pose_micro_ppm(self.fr[self.rows[i]], self.cfg, self.rank), self.cfg)


def rank_rows(n_rows, local, lw=8):
    """THIS rank's fresh rows in order: host rows j = local, local + lw, local + 2 lw, ... (never repeated)."""
    return np.arange(int(local), int(n_rows), int(lw), dtype=np.int64)


def append_stream(n_rows, local, lw=8, npm=8, sf=0, tail=1, ends=()):
    """10-22 appended plans: THIS rank's combined fresh stream over plans [1, 2, ..] (n_rows = this host's row count of each plan). Plan 1's
    rank_rows, then plan 2's, then plan 3's ...; tail=0 (the trainer's PPM_FRESH_POSE_TAIL=0): before plan k + 1 the stream is cut at
    (ends[k - 1] - sf) x npm rows, ends[k - 1] = the run's end over plans 1..k (min over ranks; the caller reduces it); tail=1: plain
    concatenation (every rank finishes its own rows first). Ids: plan 1 rows plain (= rank_rows: every step before the plan-1 end reads
    exactly what it read before), plan k + 1 rows (k << PLAN_SHIFT) | row. Each row at most once."""
    s = rank_rows(n_rows[0], local, lw)
    for k in range(1, len(n_rows)):
        if not tail:
            assert len(ends) >= k and int(ends[k - 1]) >= sf, (ends, k)
            s = s[:(int(ends[k - 1]) - int(sf)) * int(npm)]
        s = np.concatenate([s, rank_rows(n_rows[k], local, lw) | (np.int64(k) << PLAN_SHIFT)])
    return s


def append_ends(n_rows_by_host, lw=8, npm=8, sf=0, tail=1):
    """every rank at once (dry run / tests; the trainer reduces the same boundaries with all_reduce MIN): {host: [rows of plan 1, 2, ..]} ->
    ends [end over plan 1, end over plans 1..2, ..] (sf + min over ranks of len(stream) // npm) and {(host, local): stream}."""
    hosts, K = sorted(n_rows_by_host), len(next(iter(n_rows_by_host.values())))
    ends, streams = [], {}
    for k in range(1, K + 1):
        streams = {(h, l): append_stream(n_rows_by_host[h][:k], l, lw, npm, sf, tail, ends) for h in hosts for l in range(lw)}
        ends.append(int(sf) + min(len(v) // int(npm) for v in streams.values()))
    return ends, streams


def split_row(x):
    """stream id -> (plan index k: 0 = plan 1, row in that plan)."""
    return int(x) >> PLAN_SHIFT, int(x) & ((1 << PLAN_SHIFT) - 1)


class FreshPoseMultiDS:
    """FreshPoseDS over appended plans (stream ids of append_stream): item i = packed(pose_micro_ppm(plan k's row r))."""

    def __init__(self, plans, rows, cfg, rank):
        self.plans, self.rows, self.cfg, self.rank, self.fr = list(plans), [int(r) for r in rows], dict(cfg), int(rank), None
        assert len({p.get("key_prefix", "npl") for p in self.plans}) == len(self.plans), "appended plans need distinct row-key prefixes (FRESH_TAG)"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import data_ppm as D                                    # the trainer's module (sys.modules)
        if self.fr is None:
            self.fr = [None] * len(self.plans)
        k, r = split_row(self.rows[i])
        if self.fr[k] is None:
            self.fr[k] = FreshRows(self.plans[k], cache_files=48)
        return D.packed(D.pose_micro_ppm(self.fr[k][r], self.cfg, self.rank), self.cfg)


# ------------------------------------------------------------------------------------------------ check (raw parquet vs built rows / micros)
def cmd_check(a):
    import pyarrow.parquet as pq
    sys.path.insert(0, "/data/home/amartyavarma/ppm/code-811")
    import data_ppm as D  # noqa: E402
    plan = load_plan(OUT)
    fr = FreshRows(plan)
    tok = B.tok()
    z = plan["z"]
    out = {"host": HOST, "rows": []}
    pick = [0, len(fr) // 2, len(fr) - 1] if a.rows == "" else [int(x) for x in a.rows.split(",")]
    norm = lambda s: "".join(s.split())  # noqa: E731
    for i in pick:
        t0 = time.time()
        row = fr[i]
        dt = time.time() - t0
        segs = z["row_seg"][z["row_off"][i]:z["row_off"][i + 1]]
        chk = {"row": i, "build_s": round(dt, 3), "n_frames": row["n_frames"], "n_clips": len(row["clips"]), "clips_ok": 0, "word_mismatch_clips": 0}
        for c_i, (j, cr) in enumerate(zip(segs, row["clips"])):
            raw = pq.read_table(f"{plan['src_root']}/{plan['files'][int(z['seg_file'][j])]}", columns=COLS).slice(int(z["seg_row"][j]), 1).to_pylist()[0]
            P, A = raw["pose"], raw["audio"]
            b, n = cr["start"], cr["n"]
            mimi = np.asarray(A["mimi_codes"], np.int64); pose = np.asarray([P[f"tokens_cb{q}"] for q in range(32)], np.int64)
            assert mimi.shape == (32, n) and pose.shape == (32, n)
            assert np.array_equal(row["codes"][:, b:b + n].astype(np.int64), mimi), ("mimi codes differ", i, c_i)
            assert np.array_equal(row["pose"][:, b:b + n].astype(np.int64), pose), ("pose codes differ", i, c_i)
            assert cr["segment_id"] == raw["segment_id"] and cr["video_id"] == raw["video_id"]
            assert (row["kind"][cr["slot"]] == B.KIND_SLOT) and (row["kind"][b:b + n] == B.KIND_CLIP).all()
            rw = [(str(w).strip(), max(0.0, float(s))) for w, s in zip(A["word_texts"], A["word_start_seconds"]) if str(w or "").strip() and s is not None and -0.5 <= float(s) < n / B.FPS]
            rw.sort(key=lambda x: x[1])
            lane = row["lane"][b:b + n]
            txt = tok.decode([int(x) for x in lane if x < B.CTRL_LO])
            raw_txt = " ".join(w for w, _ in rw)
            if norm(txt) != norm(raw_txt):
                assert cr["words_dropped"] or cr["pieces_dropped"], ("words differ", i, c_i, txt[:200], raw_txt[:200])
                chk["word_mismatch_clips"] += 1                     # only words cut at the clip end (counted by the builder)
            # word timing: the first piece of raw word w sits at frame max(floor(start * 12.5), previous write + 1)
            if rw and (lane < B.CTRL_LO).any():
                fst = int(np.flatnonzero(lane < B.CTRL_LO)[0])
                assert fst == B.frame_of(rw[0][1]), ("first word frame", i, c_i, fst, rw[0])
            if c_i == 0:
                chk["clip0"] = {"segment_id": cr["segment_id"], "n": n, "raw_first_words": rw[:6],
                                "lane_first_word_frame": int(np.flatnonzero(lane < B.CTRL_LO)[0]) if (lane < B.CTRL_LO).any() else -1,
                                "raw_first_word_frame": B.frame_of(rw[0][1]) if rw else -1, "lane_text_head": txt[:160],
                                "new_word_frames_head": [int(q) for q in np.flatnonzero(lane == B.NW_ID)[:8]],
                                "mimi_cb0_head": mimi[0, :6].tolist(), "pose_cb0_head": pose[0, :6].tolist(), "frame0_ok": cr["frame0_ok"]}
            chk["clips_ok"] += 1
        m = D.packed(D.pose_micro_ppm(row, D.default_cfg(), 0), D.default_cfg())
        N = len(m["pid"])
        assert m["ids"].shape == (2, N) and m["depth"].shape == (2, N, 32) and m["pdepth"].shape == (N, 32)
        assert (m["pdepth"][m["pdepth"] >= 0] < 32 * 2048).all() and (m["depth"][1][m["depth"][1] >= 0] < 32 * 2048).all()
        assert int(m["ids"].max()) <= D.MAX_ID and (m["present"][0] == 0).all() and (m["present"][1] == 1).all()
        chk["micro"] = {"N": int(N), "segs": m["segs"], "n_text": m["n_text"], "words": int(m["word"].sum()), "nw_targets": int(m["nw"].sum()),
                        "pad_targets": int(m["pad"].sum()), "cb0": int((m["cb0"] >= 0).sum()), "pcb0": int((m["pcb0"] >= 0).sum()),
                        "light": int(len(m["light"])), "plight": int(len(m["plight"]))}
        out["rows"].append(chk)
        log(json.dumps(chk)[:1500])
    os.makedirs(f"{SCAN}/checks", exist_ok=True)
    json.dump(out, open(f"{SCAN}/checks/CHECK-{HOST}.json", "w"), indent=1)
    log("check OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "plan", "all", "check"])
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--rows", default="")
    a = ap.parse_args()
    if a.cmd in ("scan", "all"):
        cmd_scan(a)
    if a.cmd in ("plan", "all"):
        cmd_plan(a)
    if a.cmd == "check":
        cmd_check(a)


if __name__ == "__main__":
    main()
