#!/usr/bin/env python3
"""
download_stage2_pose_mimi_sam.py -- standalone downloader for the PRIVATE Hugging Face dataset
  InternalCan/stage2-handover-pose-mimi-sam-v1            (one parquet row per two-person conversation: tokens + metadata)
and its companion PRIVATE bucket folder
  hf://buckets/InternalCan/aloe-conversationa-data/stage2-handover-pose-mimi-sam-v1/<session>/<seg>/   (raw FLAC lanes + v1 poses)

Dependencies (Python >= 3.9):   pip install -U "huggingface_hub[hf_xet]>=2.2" pyarrow numpy
  (tokens-only runs work with any huggingface_hub >= 0.26; --audio / --poses need the bucket API of huggingface_hub >= 2.2 + hf_xet)
Auth: `export HF_TOKEN=hf_...`, or --token, or `hf auth login` (the token must read the InternalCan org; the dataset and bucket are private).

Selection rule (the colleague's rule):
  pose_both_ok   (with --include-flagged: pose_both_ok_or_flagged)   AND   crossbleed_side in {none, SPEAKER_00, SPEAKER_01}
  (`both` and `unknown` are excluded; the NAMED crossbleed side is the one that BLEEDS, so crossbleed_side == "SPEAKER_01" means
   speaker_00 is clean and speaker_01 has the partner's voice bleeding into its lane).

Output layout under --out:
  both_clean/                          crossbleed_side == none
  speaker_00_clean_speaker_01_bleeds/  crossbleed_side == SPEAKER_01
  speaker_01_clean_speaker_00_bleeds/  crossbleed_side == SPEAKER_00
    without_bands/ | with_bands/        a conversation is "with_bands" if either speaker has a no-detection interval
                                        (black missing-upload chunk) of at least --band-min-seconds (default 1.0 s)
      <session_id>/<seg>/
        conversation.json                        all scalar columns of the row (both speakers) + file list
        SPEAKER_0N.pose_tokens.npy               int16 [32, T]   (v1 pose tokens, 12.5 Hz)
        SPEAKER_0N.mimi_tokens.npy               int16 [32, T]   (mimi-stream-v5 of the raw lane)
        SPEAKER_0N.mimi_sam_isolated_tokens.npy  int16 [32, T]   (mimi-stream-v5 of the SAM-isolated stem)
        SPEAKER_0N.frame0_pose.npy               float32 [133, 3] (first pose tick: y, x, conf in source pixels)
        SPEAKER_0N.global_state0.npy             float32 [3]
        SPEAKER_0N.words.json                    [{"word","start","end"}, ...]  clip-relative seconds
        black_bands.json                         present whenever either speaker has ANY no-detection interval
        SPEAKER_0N.flac                          (--audio)  48 kHz PCM16 original lane, sha256-verified
        SPEAKER_0N/poses_25fps.npy, metadata.json, timeline.npz, job.json   (--poses)  v1 pose outputs, poses sha256-verified
Token index j (pose / Mimi / SAM streams alike) covers [0.08 j, 0.08 j + 0.08) seconds of the clip; pose tick k of poses_25fps.npy
covers [0.04 k, 0.04 k + 0.04) s (2 ticks per token).

Typical use:
  python download_stage2_pose_mimi_sam.py --out ./s2 --dry-run                      # hours table, nothing big downloaded
  python download_stage2_pose_mimi_sam.py --out ./s2                                # tokens, without_bands conversations only
  python download_stage2_pose_mimi_sam.py --out ./s2 --with-bands --what all        # + with_bands conversations, FLAC and poses
Re-running is cheap: conversations whose files are present with matching sizes are skipped.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_VERSION = "1.0 (2026-10-29)"
REPO = "InternalCan/stage2-handover-pose-mimi-sam-v1"
BUCKET = "InternalCan/aloe-conversationa-data"
BUCKET_FOLDER = "stage2-handover-pose-mimi-sam-v1"
INDEX_PATH = "index/conversations.parquet"
SHARD_INDEX_PATH = "tools/shard_index.parquet"
SPK = ("SPEAKER_00", "SPEAKER_01")
TOKEN_STREAMS = ("pose_tokens", "mimi_tokens", "mimi_sam_isolated_tokens")
TOKEN_SEC = 0.08          # one Mimi / pose-token frame (12.5 Hz)
POSE_FPS = 25             # poses_25fps.npy ticks per second (2 per token)
CONF_THRESHOLD = 0.3      # v1 definition of "no detection": all 133 joints have conf < 0.3
POSE_FILES = ("poses_25fps.npy", "metadata.json", "timeline.npz", "job.json")
FOLDER_OF_SIDE = {"none": "both_clean", "SPEAKER_01": "speaker_00_clean_speaker_01_bleeds", "SPEAKER_00": "speaker_01_clean_speaker_00_bleeds"}
FOLDERS = ("both_clean", "speaker_00_clean_speaker_01_bleeds", "speaker_01_clean_speaker_00_bleeds")
BANDS = ("without_bands", "with_bands")
# row columns that get their own file instead of going into conversation.json
ARRAY_SUFFIXES = TOKEN_STREAMS + ("frame0_pose", "global_state0", "words", "no_detection_intervals")

_print_lock = threading.Lock()
T0 = time.time()


def log(msg):
    with _print_lock:
        print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------------------------- args
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--dry-run", action="store_true", help="only fetch the small index files and print the hours table / plan")
    ap.add_argument("--limit", type=int, default=None, help="first N conversations per output folder (each <folder>/<bands> leaf); for testing")
    ap.add_argument("--with-bands", action="store_true", help="also download conversations with a black band >= --band-min-seconds (default: skip + count them)")
    ap.add_argument("--band-min-seconds", type=float, default=1.0, help="a no-detection interval at least this long makes a conversation 'with_bands' (default 1.0)")
    ap.add_argument("--include-flagged", action="store_true", help="select pose_both_ok_or_flagged instead of pose_both_ok")
    ap.add_argument("--what", choices=("tokens", "tokens+audio", "all"), default="tokens", help="tokens (default) | tokens+audio | all (= tokens + audio + poses)")
    ap.add_argument("--audio", action="store_true", help="also download SPEAKER_00.flac / SPEAKER_01.flac from the bucket (sha256-verified)")
    ap.add_argument("--poses", action="store_true", help="also download SPEAKER_0N/{poses_25fps.npy,metadata.json,timeline.npz,job.json} (poses sha256-verified)")
    ap.add_argument("--workers", type=int, default=8, help="parallel parquet shard downloads (default 8)")
    ap.add_argument("--bucket-batch", type=int, default=64, help="bucket files per download_bucket_files call (default 64)")
    ap.add_argument("--cache", default=None, help="cache dir for parquet shards / index files (default <out>/.cache)")
    ap.add_argument("--keep-cache", action="store_true", help="keep the parquet shards after extracting (default: delete each shard once used)")
    ap.add_argument("--verify-existing", action="store_true", help="re-hash FLAC / poses files that are already present (default: trust size match)")
    ap.add_argument("--rescan", action="store_true", help="ignore tools/shard_index.parquet and rebuild the shard index by scanning the shards (slow, ~5 min)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--token", default=None, help="HF token (default: $HF_TOKEN, then the token saved by `hf auth login`)")
    ap.add_argument("--progress", action="store_true", help="show huggingface_hub progress bars")
    a = ap.parse_args()
    if a.what in ("tokens+audio", "all"):
        a.audio = True
    if a.what == "all":
        a.poses = True
    a.out = os.path.abspath(a.out)
    a.cache = os.path.abspath(a.cache) if a.cache else os.path.join(a.out, ".cache")
    return a


def resolve_token(a):
    tok = a.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        try:
            from huggingface_hub import get_token
            tok = get_token()
        except Exception:
            tok = None
    if not tok:
        sys.exit("No Hugging Face token: set HF_TOKEN, pass --token, or run `hf auth login` (the dataset and bucket are private).")
    return tok


def sanitize(v):
    """Make a pyarrow .as_py() value JSON-safe (NaN -> null)."""
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, dict):
        return {k: sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [sanitize(x) for x in v]
    return v


def sha256_file(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def retry(fn, what, tries=6):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = min(60, 3 * 2 ** attempt)
            log(f"retry {attempt + 1}/{tries - 1} for {what} in {wait}s: {type(e).__name__}: {str(e)[:200]}")
            time.sleep(wait)


# ----------------------------------------------------------------------------------------------------------------------- index
def hf_get(a, token, path_in_repo, force=False):
    """Download one small repo file into the cache dir; returns the local path."""
    from huggingface_hub import hf_hub_download
    return retry(lambda: hf_hub_download(a.repo, path_in_repo, repo_type="dataset", revision=a.revision, local_dir=a.cache, token=token,
                                         force_download=force), f"download {path_in_repo}")


def scan_shards(a, token):
    """Fallback when tools/shard_index.parquet is missing: read only the key + no_detection_intervals columns of every shard
    (HTTP range reads through HfFileSystem, ~5 min for 720 shards). Returns a pyarrow table."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem
    api = HfApi(token=token)
    shards = sorted(f for f in api.list_repo_files(a.repo, repo_type="dataset", revision=a.revision) if f.startswith("data/") and f.endswith(".parquet"))
    fs = HfFileSystem(token=token)
    cols = ["session_id", "seg", "T", "speaker_00_no_detection_intervals", "speaker_01_no_detection_intervals"]
    log(f"scanning {len(shards)} shards for keys + no-detection intervals (slow fallback; use tools/shard_index.parquet when available)")

    def one(shard):
        def go():
            with fs.open(f"datasets/{a.repo}@{a.revision}/{shard}", "rb", block_size=256 * 1024) as f:
                t = pq.ParquetFile(f).read(columns=cols)
            t = t.append_column("shard", pa.array([shard] * t.num_rows, pa.string()))
            return t.append_column("row_index", pa.array(list(range(t.num_rows)), pa.int32()))
        return retry(go, f"scan {shard}")

    tables = []
    with ThreadPoolExecutor(max(a.workers, 8)) as ex:
        futs = [ex.submit(one, s) for s in shards]
        for i, fu in enumerate(as_completed(futs)):
            tables.append(fu.result())
            if (i + 1) % 100 == 0:
                log(f"  scanned {i + 1}/{len(shards)} shards")
    t = pa.concat_tables(tables).sort_by([("shard", "ascending"), ("row_index", "ascending")])
    os.makedirs(os.path.join(a.cache, "tools"), exist_ok=True)
    pq.write_table(t, os.path.join(a.cache, SHARD_INDEX_PATH), compression="zstd")
    return t


def load_shard_index(a, token):
    import pyarrow.parquet as pq
    from huggingface_hub.errors import EntryNotFoundError
    if not a.rescan:
        try:
            return pq.read_table(hf_get(a, token, SHARD_INDEX_PATH))
        except EntryNotFoundError:
            log(f"{SHARD_INDEX_PATH} not in the repo -> scanning shards")
        except Exception as e:  # noqa: BLE001
            log(f"could not fetch {SHARD_INDEX_PATH} ({type(e).__name__}: {str(e)[:200]}) -> scanning shards")
    return scan_shards(a, token)


def tick(x):
    """seconds -> 25 fps tick index (intervals are tick-aligned k/25 but stored as float32)."""
    return int(round(x * POSE_FPS))


def band_info(intervals, T):
    """Per-speaker black-band summary from the row's no_detection_intervals ([start_s, end_s] pairs, end exclusive)."""
    ivs = []
    for s, e in intervals or []:
        ts, te = tick(s), tick(e)
        ivs.append({"start_s": round(ts / POSE_FPS, 3), "end_s": round(te / POSE_FPS, 3), "seconds": round((te - ts) / POSE_FPS, 3),
                    "pose_ticks": [ts, min(te, 2 * T)], "tokens": [ts // 2, min(-(-te // 2), T)]})
    return {"intervals": [[iv["start_s"], iv["end_s"]] for iv in ivs], "seconds": round(sum(iv["seconds"] for iv in ivs), 3), "n": len(ivs),
            "max_seconds": max([iv["seconds"] for iv in ivs], default=0.0),
            "pose_tick_intervals": [iv["pose_ticks"] for iv in ivs], "token_intervals": [iv["tokens"] for iv in ivs]}


def max_gap(intervals):
    return max([(tick(e) - tick(s)) / POSE_FPS for s, e in (intervals or [])], default=0.0)


class Conv:
    __slots__ = ("key", "session_id", "seg", "folder", "bands", "T", "hours", "shard", "intervals", "max_gap", "audio_bytes", "poses_bytes",
                 "audio_sha", "poses_sha", "audio_path", "poses_path", "pose_dir", "dir", "status", "errors")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        self.errors = []


def select_conversations(a, index_table, shard_table):
    """Apply the selection rule + band classification to the index; returns list[Conv] (sorted by session/seg)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    idx = index_table
    ok_col = "pose_both_ok_or_flagged" if a.include_flagged else "pose_both_ok"
    mask = pc.and_(pc.fill_null(idx[ok_col], False), pc.is_in(idx["crossbleed_side"], value_set=pa.array(list(FOLDER_OF_SIDE))))
    sel = idx.filter(mask)
    shard_rows = {}
    for r in shard_table.select(["session_id", "seg", "shard", "speaker_00_no_detection_intervals", "speaker_01_no_detection_intervals"]).to_pylist():
        shard_rows[f"{r['session_id']}/{r['seg']}"] = r
    convs, missing = [], []
    cols = ["session_id", "seg", "T", "clip_duration_sec", "crossbleed_side"] + \
           [f"{s.lower()}_{c}" for s in SPK for c in ("audio_bytes", "poses_bytes", "audio_sha256", "poses_sha256", "audio_hf_path", "poses_hf_path", "pose_dir_hf_path")]
    for r in sel.select(cols).to_pylist():
        key = f"{r['session_id']}/{r['seg']}"
        sr = shard_rows.get(key)
        if sr is None:
            missing.append(key)
            continue
        intervals = {s: (sr[f"{s.lower()}_no_detection_intervals"] or []) for s in SPK}
        mg = max(max_gap(intervals[s]) for s in SPK)
        bands = "with_bands" if mg >= a.band_min_seconds else "without_bands"
        folder = FOLDER_OF_SIDE[r["crossbleed_side"]]
        convs.append(Conv(key=key, session_id=r["session_id"], seg=r["seg"], folder=folder, bands=bands, T=int(r["T"]), hours=float(r["clip_duration_sec"]) / 3600.0,
                          shard=sr["shard"], intervals=intervals, max_gap=mg,
                          audio_bytes={s: r[f"{s.lower()}_audio_bytes"] for s in SPK}, poses_bytes={s: r[f"{s.lower()}_poses_bytes"] for s in SPK},
                          audio_sha={s: r[f"{s.lower()}_audio_sha256"] for s in SPK}, poses_sha={s: r[f"{s.lower()}_poses_sha256"] for s in SPK},
                          audio_path={s: r[f"{s.lower()}_audio_hf_path"] for s in SPK}, poses_path={s: r[f"{s.lower()}_poses_hf_path"] for s in SPK},
                          pose_dir={s: (r[f"{s.lower()}_pose_dir_hf_path"] or "").rstrip("/") for s in SPK},
                          dir=os.path.join(a.out, folder, bands, r["session_id"], r["seg"]), status="planned"))
    convs.sort(key=lambda c: (c.folder, c.bands, c.session_id, c.seg))
    if missing:
        log(f"WARNING: {len(missing)} selected conversations are not in the shard index (first: {missing[:3]}); run with --rescan")
    return convs, missing


def hours_table(convs, a):
    """Rows of the table printed by --dry-run and at the end: per folder x bands counts, hours, bytes."""
    rows = []
    for folder in FOLDERS:
        for bands in BANDS:
            cs = [c for c in convs if c.folder == folder and c.bands == bands]
            rows.append({"folder": folder, "bands": bands, "conversations": len(cs), "hours": round(sum(c.hours for c in cs), 2),
                         "audio_gb": round(sum((c.audio_bytes[s] or 0) for c in cs for s in SPK) / 1e9, 2),
                         "poses_gb": round(sum((c.poses_bytes[s] or 0) for c in cs for s in SPK) / 1e9, 2),
                         "tokens_gb": round(sum(6 * 32 * c.T * 2 for c in cs) / 1e9, 2),
                         "download": (bands == "without_bands") or a.with_bands})
    return rows


def print_table(rows, convs, a, planned, skipped_bands):
    w = max(len(r["folder"]) for r in rows)
    hdr = f"{'folder':{w}}  {'bands':13} {'convs':>6} {'hours':>8} {'tokens_GB':>9} {'audio_GB':>9} {'poses_GB':>9}  action"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        act = "download" if r["download"] else "SKIP (add --with-bands)"
        print(f"{r['folder']:{w}}  {r['bands']:13} {r['conversations']:6d} {r['hours']:8.2f} {r['tokens_gb']:9.2f} {r['audio_gb']:9.2f} {r['poses_gb']:9.2f}  {act}")
    tot_h = sum(c.hours for c in convs)
    print("-" * len(hdr))
    print(f"{'TOTAL selected':{w}}  {'':13} {len(convs):6d} {tot_h:8.2f}")
    wb = [c for c in convs if c.bands == "with_bands"]
    print(f"  with_bands (>= {a.band_min_seconds:g} s black chunk in either speaker): {len(wb)} conversations / {sum(c.hours for c in wb):.2f} h")
    what = "tokens" + (" + audio (FLAC)" if a.audio else "") + (" + poses" if a.poses else "")
    gb = sum(6 * 32 * c.T * 2 for c in planned) / 1e9 + (sum((c.audio_bytes[s] or 0) for c in planned for s in SPK) / 1e9 if a.audio else 0) + \
        (sum((c.poses_bytes[s] or 0) for c in planned for s in SPK) / 1e9 if a.poses else 0)
    lim = f", --limit {a.limit} per output folder" if a.limit else ""
    print(f"  plan with the current flags ({what}{lim}): {len(planned)} conversations / {sum(c.hours for c in planned):.2f} h, ~{gb:.1f} GB on disk"
          f"; skipped because with_bands: {skipped_bands}")
    print(f"  parquet shards to fetch: {len(set(c.shard for c in planned))} (deleted after use unless --keep-cache)\n")


# ------------------------------------------------------------------------------------------------------------------ per-conversation
def token_files(c):
    return [f"{s}.{st}.npy" for s in SPK for st in TOKEN_STREAMS] + [f"{s}.frame0_pose.npy" for s in SPK] + [f"{s}.global_state0.npy" for s in SPK] + [f"{s}.words.json" for s in SPK]


def bucket_files(c, a):
    """[(remote_path_in_bucket, relative_local_name, expected_bytes or None, expected_sha256 or None)] for the current flags."""
    out = []
    if a.audio:
        for s in SPK:
            out.append((c.audio_path[s], f"{s}.flac", c.audio_bytes[s], c.audio_sha[s]))
    if a.poses:
        for s in SPK:
            for fn in POSE_FILES:
                if fn == "poses_25fps.npy":
                    out.append((c.poses_path[s], f"{s}/{fn}", c.poses_bytes[s], c.poses_sha[s]))
                else:
                    out.append((f"{c.pose_dir[s]}/{fn}", f"{s}/{fn}", None, None))
    return out


def read_conv_json(c):
    p = os.path.join(c.dir, "conversation.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def present(path, size):
    return size is not None and os.path.isfile(path) and os.path.getsize(path) == size


def tokens_done(c, cj):
    """conversation.json exists and every token-stage file it lists is present with the recorded size."""
    if not cj or cj.get("_downloader", {}).get("tokens_complete") is not True:
        return False
    files = cj.get("_files", {})
    need = token_files(c) + (["black_bands.json"] if any(c.intervals[s] for s in SPK) else [])
    return all(f in files and present(os.path.join(c.dir, f), files[f]) for f in need)


def missing_bucket_files(c, a, cj):
    files = (cj or {}).get("_files", {})
    out = []
    for remote, rel, nbytes, sha in bucket_files(c, a):
        local = os.path.join(c.dir, rel)
        expected = nbytes if nbytes is not None else files.get(rel)
        if present(local, expected):
            if a.verify_existing and sha and sha256_file(local) != sha:
                out.append((remote, rel, nbytes, sha))
            continue
        out.append((remote, rel, nbytes, sha))
    return out


def list2d_to_numpy(scalar, dtype):
    """pyarrow ListScalar of list<list<x>> -> 2-D numpy array (rows must have equal length)."""
    import numpy as np
    inner = scalar.values
    lengths = inner.value_lengths().to_numpy(zero_copy_only=False)
    if len(lengths) == 0:
        return np.zeros((0, 0), dtype=dtype)
    if not (lengths == lengths[0]).all():
        raise ValueError(f"ragged 2-D list: lengths {sorted(set(lengths.tolist()))}")
    flat = inner.flatten().to_numpy(zero_copy_only=False)
    return np.ascontiguousarray(flat.reshape(len(lengths), int(lengths[0])).astype(dtype, copy=False))


def write_conversation(table, i, c, a):
    """Write the token-stage files of conversation c from row i of a shard table. Returns the list of errors (empty = ok)."""
    import numpy as np
    errors = []
    os.makedirs(c.dir, exist_ok=True)
    row_T = table["T"][i].as_py()
    if row_T != c.T:
        errors.append(f"T mismatch index {c.T} vs row {row_T}")
    T = int(row_T)
    files = {}
    # --- scalars -> conversation.json
    scalars = {}
    for name in table.column_names:
        if any(name == f"{s.lower()}_{suf}" for s in SPK for suf in ARRAY_SUFFIXES):
            continue
        scalars[name] = sanitize(table[name][i].as_py())
    # --- arrays
    shapes = {}
    for s in SPK:
        pre = s.lower() + "_"
        for st in TOKEN_STREAMS:
            col = f"{pre}{st}"
            sc = table[col][i]
            if not sc.is_valid:
                errors.append(f"{col} is null")
                continue
            try:
                arr = list2d_to_numpy(sc, np.int16)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{col}: {e}")
                continue
            if arr.shape != (32, T):
                errors.append(f"{col} shape {arr.shape} != (32, {T})")
            shapes[col] = arr.shape
            fn = f"{s}.{st}.npy"
            np.save(os.path.join(c.dir, fn), arr)
            files[fn] = os.path.getsize(os.path.join(c.dir, fn))
        sc = table[f"{pre}frame0_pose"][i]
        if sc.is_valid:
            f0 = list2d_to_numpy(sc, np.float32)
            if f0.shape != (133, 3):
                errors.append(f"{pre}frame0_pose shape {f0.shape} != (133, 3)")
            np.save(os.path.join(c.dir, f"{s}.frame0_pose.npy"), f0)
            files[f"{s}.frame0_pose.npy"] = os.path.getsize(os.path.join(c.dir, f"{s}.frame0_pose.npy"))
        else:
            errors.append(f"{pre}frame0_pose is null")
        sc = table[f"{pre}global_state0"][i]
        if sc.is_valid:
            g0 = np.asarray(sc.as_py(), dtype=np.float32)
            if g0.shape != (3,):
                errors.append(f"{pre}global_state0 shape {g0.shape} != (3,)")
            np.save(os.path.join(c.dir, f"{s}.global_state0.npy"), g0)
            files[f"{s}.global_state0.npy"] = os.path.getsize(os.path.join(c.dir, f"{s}.global_state0.npy"))
        else:
            errors.append(f"{pre}global_state0 is null")
        words = table[f"{pre}words"][i].as_py() or []
        with open(os.path.join(c.dir, f"{s}.words.json"), "w") as fh:
            json.dump(sanitize(words), fh)
        files[f"{s}.words.json"] = os.path.getsize(os.path.join(c.dir, f"{s}.words.json"))
    # --- black bands (from the row itself; the index copy was used for the folder decision)
    row_iv = {s: (table[f"{s.lower()}_no_detection_intervals"][i].as_py() or []) for s in SPK}
    if any(row_iv[s] for s in SPK):
        bb = {"definition": f"25 fps ticks of the v1 poses_25fps.npy where all 133 joints have conf < {CONF_THRESHOLD} (black missing-upload chunks; "
                            f"v1 pose_tokens hold the last confident pose through them -> mask or drop for training)",
              "conf_threshold": CONF_THRESHOLD, "band_min_seconds_used": a.band_min_seconds, "with_bands": c.bands == "with_bands",
              "time_base": "clip-relative seconds; token j covers [0.08 j, 0.08 j + 0.08) s, pose tick k covers [0.04 k, 0.04 k + 0.04) s; "
                           "token_intervals = [floor(start / 0.08), ceil(end / 0.08)) clipped to T, pose_tick_intervals = [start * 25, end * 25)",
              "T": T, "pose_frames": 2 * T, "speakers": {s: band_info(row_iv[s], T) for s in SPK}}
        with open(os.path.join(c.dir, "black_bands.json"), "w") as fh:
            json.dump(bb, fh, indent=1)
        files["black_bands.json"] = os.path.getsize(os.path.join(c.dir, "black_bands.json"))
    # --- T consistency across the (up to) 6 streams
    Ts = {k: v[1] for k, v in shapes.items()}
    if len(set(Ts.values())) > 1:
        errors.append(f"T differs across streams: {Ts}")
    scalars["_downloader"] = {"script_version": SCRIPT_VERSION, "repo": a.repo, "revision": a.revision, "shard": c.shard, "output_folder": c.folder,
                              "bands": c.bands, "band_min_seconds": a.band_min_seconds, "max_no_detection_seconds": round(c.max_gap, 3),
                              "token_frame_seconds": TOKEN_SEC, "pose_fps": POSE_FPS, "tokens_complete": not errors, "errors": errors,
                              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    scalars["_files"] = files
    tmp = os.path.join(c.dir, "conversation.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(scalars, fh, indent=1)
    os.replace(tmp, os.path.join(c.dir, "conversation.json"))
    return errors


def update_conv_files(c, new_files):
    cj = read_conv_json(c)
    if cj is None:
        return
    cj.setdefault("_files", {}).update(new_files)
    tmp = os.path.join(c.dir, "conversation.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(cj, fh, indent=1)
    os.replace(tmp, os.path.join(c.dir, "conversation.json"))


# ------------------------------------------------------------------------------------------------------------------------ stages
def stage_tokens(a, token, need):
    """Download the shards that hold the conversations in `need` (key -> Conv) in parallel and write their token files."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    by_shard = {}
    for c in need.values():
        by_shard.setdefault(c.shard, []).append(c)
    log(f"tokens: {len(need)} conversations from {len(by_shard)} parquet shards, {a.workers} workers")
    done_shards = 0
    done_convs = 0

    def one(shard, convs):
        local = retry(lambda: hf_hub_download(a.repo, shard, repo_type="dataset", revision=a.revision, local_dir=a.cache, token=token), f"download {shard}")
        t = pq.read_table(local)
        keys = [f"{s}/{g}" for s, g in zip(t["session_id"].to_pylist(), t["seg"].to_pylist())]
        pos = {k: i for i, k in enumerate(keys)}
        res = []
        for c in convs:
            i = pos.get(c.key)
            if i is None:
                c.errors.append(f"row not found in {shard} (stale shard index? try --rescan)")
                c.status = "failed"
                res.append(c)
                continue
            try:
                errs = write_conversation(t, i, c, a)
            except Exception as e:  # noqa: BLE001
                errs = [f"exception writing files: {type(e).__name__}: {e}"]
            c.errors.extend(errs)
            c.status = "failed" if errs else "tokens_ok"
            res.append(c)
        del t
        if not a.keep_cache:
            try:
                os.remove(local)
            except OSError:
                pass
        return res

    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, s, cs): s for s, cs in by_shard.items()}
        for fu in as_completed(futs):
            shard = futs[fu]
            try:
                res = fu.result()
            except Exception as e:  # noqa: BLE001
                for c in by_shard[shard]:
                    c.errors.append(f"shard {shard}: {type(e).__name__}: {str(e)[:300]}")
                    c.status = "failed"
                res = by_shard[shard]
            done_shards += 1
            done_convs += len(res)
            if done_shards % 20 == 0 or done_shards == len(by_shard):
                log(f"  shards {done_shards}/{len(by_shard)}, conversations {done_convs}/{len(need)}")


def stage_bucket(a, token, items):
    """items: [(Conv, remote, rel, nbytes, sha)]. Downloads in batches through HfApi.download_bucket_files, verifies size + sha256."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    if not hasattr(api, "download_bucket_files"):
        import huggingface_hub
        sys.exit(f"--audio/--poses need huggingface_hub >= 2.2 with hf_xet (installed: {huggingface_hub.__version__}): "
                 f"pip install -U 'huggingface_hub[hf_xet]>=2.2'")
    total_bytes = sum((it[3] or 0) for it in items)
    log(f"bucket: {len(items)} files (~{total_bytes / 1e9:.1f} GB known sizes) from hf://buckets/{a.bucket}/ in batches of {a.bucket_batch}")
    got_bytes = 0
    for b0 in range(0, len(items), a.bucket_batch):
        batch = items[b0:b0 + a.bucket_batch]
        pairs = []
        for c, remote, rel, nbytes, sha in batch:
            local = os.path.join(c.dir, rel)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            pairs.append((remote, local + ".part"))
        try:
            retry(lambda: api.download_bucket_files(a.bucket, pairs, raise_on_missing_files=True), f"bucket batch {b0 // a.bucket_batch + 1}", tries=4)
            failed_batch = None
        except Exception as e:  # noqa: BLE001
            failed_batch = f"{type(e).__name__}: {str(e)[:300]}"
        per_conv = {}
        for (c, remote, rel, nbytes, sha), (_, part) in zip(batch, pairs):
            local = part[:-5]
            if failed_batch is not None and not os.path.isfile(part):
                # isolate: retry this single file once
                try:
                    api.download_bucket_files(a.bucket, [(remote, part)], raise_on_missing_files=True)
                except Exception as e:  # noqa: BLE001
                    c.errors.append(f"{rel}: download failed: {type(e).__name__}: {str(e)[:200]}")
                    continue
            if not os.path.isfile(part):
                c.errors.append(f"{rel}: not downloaded")
                continue
            size = os.path.getsize(part)
            if nbytes is not None and size != nbytes:
                c.errors.append(f"{rel}: size {size} != expected {nbytes}")
                os.remove(part)
                continue
            if sha:
                got = sha256_file(part)
                if got != sha:
                    c.errors.append(f"{rel}: sha256 {got[:16]}... != expected {sha[:16]}...")
                    os.remove(part)
                    continue
            os.replace(part, local)
            got_bytes += size
            per_conv.setdefault(c.key, (c, {}))[1][rel] = size
        for c, files in per_conv.values():
            update_conv_files(c, files)
        log(f"  bucket files {min(b0 + a.bucket_batch, len(items))}/{len(items)}, {got_bytes / 1e9:.2f} GB")


def stage_verify(a, convs):
    """Final checks on what is on disk for every planned conversation: shapes, T equality, poses frames == 2T."""
    import numpy as np
    for c in convs:
        if c.status == "failed":
            continue
        cj = read_conv_json(c)
        if cj is None:
            c.errors.append("conversation.json missing")
            c.status = "failed"
            continue
        Ts = {}
        for s in SPK:
            for st in TOKEN_STREAMS:
                p = os.path.join(c.dir, f"{s}.{st}.npy")
                if not os.path.isfile(p):
                    c.errors.append(f"{s}.{st}.npy missing")
                    continue
                arr = np.load(p, mmap_mode="r")
                if arr.ndim != 2 or arr.shape[0] != 32 or arr.dtype != np.int16:
                    c.errors.append(f"{s}.{st}.npy shape {arr.shape} dtype {arr.dtype}")
                    continue
                Ts[f"{s}.{st}"] = arr.shape[1]
            p = os.path.join(c.dir, f"{s}.frame0_pose.npy")
            if os.path.isfile(p) and np.load(p, mmap_mode="r").shape != (133, 3):
                c.errors.append(f"{s}.frame0_pose.npy bad shape")
            if a.poses:
                p = os.path.join(c.dir, s, "poses_25fps.npy")
                if os.path.isfile(p):
                    n = np.load(p, mmap_mode="r").shape[0]
                    if n != 2 * c.T:
                        c.errors.append(f"{s}/poses_25fps.npy frames {n} != 2T = {2 * c.T}")
                else:
                    c.errors.append(f"{s}/poses_25fps.npy missing")
        if Ts and (len(set(Ts.values())) != 1 or set(Ts.values()) != {c.T}):
            c.errors.append(f"T inconsistent: {Ts} (index T {c.T})")
        for remote, rel, nbytes, sha in bucket_files(c, a):
            if not os.path.isfile(os.path.join(c.dir, rel)):
                c.errors.append(f"{rel} missing")
        c.status = "failed" if c.errors else "ok"


def write_manifest(a, convs, planned_keys, skipped_bands, rows, dry_run):
    os.makedirs(a.out, exist_ok=True)
    mpath = os.path.join(a.out, "manifest.csv")
    with open(mpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["folder", "bands", "session_id", "seg", "T", "hours", "max_no_detection_sec", "planned", "status", "dir", "audio_bytes", "poses_bytes", "shard", "errors"])
        for c in convs:
            w.writerow([c.folder, c.bands, c.session_id, c.seg, c.T, f"{c.hours:.5f}", f"{c.max_gap:.2f}", int(c.key in planned_keys), c.status,
                        os.path.relpath(c.dir, a.out), sum((c.audio_bytes[s] or 0) for s in SPK), sum((c.poses_bytes[s] or 0) for s in SPK), c.shard, " | ".join(c.errors)])
    failures = [{"key": c.key, "folder": c.folder, "bands": c.bands, "errors": c.errors} for c in convs if c.errors]
    summary = {"script_version": SCRIPT_VERSION, "repo": a.repo, "revision": a.revision, "bucket": a.bucket, "dry_run": dry_run,
               "flags": {k: v for k, v in vars(a).items() if k != "token"}, "selection": {"pose_column": "pose_both_ok_or_flagged" if a.include_flagged else "pose_both_ok",
               "crossbleed_side_in": sorted(FOLDER_OF_SIDE), "band_rule": f"with_bands if either speaker has a no-detection interval >= {a.band_min_seconds} s"},
               "table": rows, "selected_conversations": len(convs), "selected_hours": round(sum(c.hours for c in convs), 3),
               "planned_conversations": len(planned_keys), "planned_hours": round(sum(c.hours for c in convs if c.key in planned_keys), 3),
               "skipped_with_bands": skipped_bands, "status_counts": {s: sum(1 for c in convs if c.status == s) for s in sorted(set(c.status for c in convs))},
               "failures": failures, "elapsed_sec": round(time.time() - T0, 1), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return failures


# -------------------------------------------------------------------------------------------------------------------------- main
def main():
    a = parse_args()
    if not a.progress:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.pop("HF_HUB_OFFLINE", None)
    import pyarrow.parquet as pq
    token = resolve_token(a)
    os.makedirs(a.cache, exist_ok=True)
    prev = os.path.join(a.out, "summary.json")
    if os.path.exists(prev):
        try:
            pb = json.load(open(prev)).get("flags", {}).get("band_min_seconds")
            if pb is not None and abs(pb - a.band_min_seconds) > 1e-9:
                log(f"WARNING: previous run in {a.out} used --band-min-seconds {pb}; using {a.band_min_seconds} changes the with/without_bands folder split")
        except Exception:  # noqa: BLE001
            pass
    log(f"downloader {SCRIPT_VERSION}; repo {a.repo}@{a.revision}; out {a.out}; cache {a.cache}")
    index = pq.read_table(hf_get(a, token, INDEX_PATH))
    shard_index = load_shard_index(a, token)
    log(f"index: {index.num_rows} conversations, shard index: {shard_index.num_rows} rows in {len(set(shard_index['shard'].to_pylist()))} shards")
    convs, missing = select_conversations(a, index, shard_index)
    rows = hours_table(convs, a)
    # ---- plan
    planned, skipped_bands, per_leaf = [], 0, {}
    for c in convs:
        if c.bands == "with_bands" and not a.with_bands:
            skipped_bands += 1
            c.status = "skipped_with_bands"
            continue
        leaf = (c.folder, c.bands)
        if a.limit is not None and per_leaf.get(leaf, 0) >= a.limit:
            c.status = "skipped_limit"
            continue
        per_leaf[leaf] = per_leaf.get(leaf, 0) + 1
        planned.append(c)
    planned_keys = {c.key for c in planned}
    print_table(rows, convs, a, planned, skipped_bands)
    if a.dry_run:
        failures = write_manifest(a, convs, planned_keys, skipped_bands, rows, dry_run=True)
        log(f"dry run: wrote {os.path.join(a.out, 'manifest.csv')} and summary.json; nothing else downloaded")
        return 1 if missing else 0
    # ---- resume status
    need_tokens, bucket_items, resumed = {}, [], 0
    for c in planned:
        cj = read_conv_json(c)
        if tokens_done(c, cj):
            c.status = "tokens_ok"
            resumed += 1
        else:
            need_tokens[c.key] = c
        for remote, rel, nbytes, sha in missing_bucket_files(c, a, cj):
            bucket_items.append((c, remote, rel, nbytes, sha))
    log(f"resume: {resumed} conversations already have their token files; {len(need_tokens)} to extract; {len(bucket_items)} bucket files to fetch")
    if need_tokens:
        stage_tokens(a, token, need_tokens)
    if bucket_items:
        stage_bucket(a, token, [it for it in bucket_items if it[0].status != "failed"])
    stage_verify(a, planned)
    failures = write_manifest(a, convs, planned_keys, skipped_bands, rows, dry_run=False)
    ok = sum(1 for c in planned if c.status == "ok")
    print()
    log(f"done: {ok}/{len(planned)} conversations complete ({sum(c.hours for c in planned if c.status == 'ok'):.2f} h), {len(failures)} with failures; "
        f"skipped with_bands {skipped_bands}; manifest {os.path.join(a.out, 'manifest.csv')}; summary.json written")
    for f in failures[:20]:
        log(f"  FAIL {f['key']} [{f['folder']}/{f['bands']}]: {'; '.join(f['errors'])[:400]}")
    if len(failures) > 20:
        log(f"  ... {len(failures) - 20} more in summary.json")
    if not a.keep_cache:
        shutil.rmtree(os.path.join(a.cache, "data"), ignore_errors=True)
    return 1 if failures or missing else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted; re-run the same command to resume", file=sys.stderr)
        sys.exit(130)
