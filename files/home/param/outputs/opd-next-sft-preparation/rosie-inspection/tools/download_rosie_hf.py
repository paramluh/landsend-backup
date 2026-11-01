#!/usr/bin/env python3
"""Unpack the private Hugging Face dataset InternalCan/rosie-stamp-conversations-v1 (ALL of Rosie Stamp's conversations: 216 / 35.0 h,
both speakers) into per-conversation folders — Hugging Face only, no GCP access needed.

    pip install -U "huggingface_hub[hf_xet]" pyarrow numpy
    hf auth login                                   # a READ token of an InternalCan member (or export HF_TOKEN=hf_...)
    hf download InternalCan/rosie-stamp-conversations-v1 --repo-type dataset --local-dir ./rosie
    python3 rosie/tools/download_rosie_hf.py --out ./rosie-data                  # tokens + labels (fast, ~0.6 GB)
    python3 rosie/tools/download_rosie_hf.py --out ./rosie-data --with-audio     # + the FLAC lanes from audio/*.tar (~11.5 GB)

Without `hf download` the script fetches what it needs itself (huggingface_hub.snapshot_download, --repo / --token / HF_TOKEN).
When run from inside a downloaded copy (rosie/tools/download_rosie_hf.py) that copy is used automatically; or pass --local-dir ./rosie.

Output  <out>/Rosie_Stamp/<session_id>/<seg>/   (same names as tools/download_actor_tokens.py, the GCS downloader; Rosie = ASSISTANT, partner = user)
  Rosie_Stamp.original.codes.npy      user.original.codes.npy       original Mimi tokens (mimi-stream-v5): int16 [32, T], 12.5 tokens/s
  Rosie_Stamp.sam_mix.codes.npy       user.sam_mix.codes.npy        SAM on the summed mixture        (set A only: stage2-handover-20261008)
  Rosie_Stamp.sam_isolated.codes.npy  user.sam_isolated.codes.npy   SAM on each lane alone           (set A only)
  words.json   vad.json   crossbleed.json   conversation.json       relabelled by role ("Rosie Stamp" | "user", lane kept in `lane`)
  _gcs/w00.mimi.json [_gcs/w00.samsep.mimi.json _gcs/sam_sep.json] _gcs/words.json _gcs/vad.json   untouched source documents (when present in the row)
  --with-audio: Rosie_Stamp.original.flac user.original.flac (48 kHz lanes) [+ Rosie_Stamp.sam_mix.flac user.sam_mix.flac
                Rosie_Stamp.sam_isolated.flac user.sam_isolated.flac — 24 kHz SAM stems, set A only]
Also: <out>/manifest.csv (one row per conversation), <out>/README.md (the dataset card), <out>/_manifests/download_summary.<ts>.json.
Every .npy is header-checked ([32, T] int16 with the row's mimi_T); every FLAC is size-checked (and crc32c-checked if google-crc32c is installed).
Resumable: existing same-size files are kept (--force rewrites). Requirements: Python 3.9+, pyarrow, huggingface_hub (numpy optional).
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = "InternalCan/rosie-stamp-conversations-v1"
ACTOR_SLUG = "Rosie_Stamp"
TOKEN_COLS = {"rosie_original_codes": "Rosie_Stamp.original.codes.npy", "user_original_codes": "user.original.codes.npy",
              "rosie_sam_mix_codes": "Rosie_Stamp.sam_mix.codes.npy", "user_sam_mix_codes": "user.sam_mix.codes.npy",
              "rosie_sam_isolated_codes": "Rosie_Stamp.sam_isolated.codes.npy", "user_sam_isolated_codes": "user.sam_isolated.codes.npy"}
JSON_COLS = {"words_json": "words.json", "vad_json": "vad.json", "crossbleed_json": "crossbleed.json", "conversation_json": "conversation.json",
             "original_mimi_json": "_gcs/w00.mimi.json", "samsep_mimi_json": "_gcs/w00.samsep.mimi.json", "sam_sep_json": "_gcs/sam_sep.json",
             "source_words_json": "_gcs/words.json", "source_vad_json": "_gcs/vad.json"}


def npy_shape(data: bytes) -> Tuple[Tuple[int, ...], str]:
    if data[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major = data[6]
    hlen = int.from_bytes(data[8:10] if major == 1 else data[8:12], "little")
    start = 10 if major == 1 else 12
    header = ast.literal_eval(data[start:start + hlen].decode("latin1").strip())
    return tuple(header["shape"]), str(header["descr"])


def write_bytes(path: Path, data: bytes, force: bool) -> str:
    if not force and path.exists() and path.stat().st_size == len(data):
        return "present"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part-{}".format(os.getpid()))
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return "new"


def locate(args: argparse.Namespace) -> Path:
    """Directory holding index/, tokens/ (and audio/ with --with-audio): --local-dir, the copy this script sits in, or snapshot_download."""
    if args.local_dir:
        root = Path(args.local_dir)
        if not (root / "index" / "conversations.parquet").exists():
            sys.exit("{} has no index/conversations.parquet (run `hf download {} --repo-type dataset --local-dir {}` first)".format(root, args.repo, root))
    else:
        here = Path(__file__).resolve().parent.parent
        if (here / "index" / "conversations.parquet").exists() and not args.fetch:
            root = here
        else:
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                sys.exit("huggingface_hub is not installed: pip install -U 'huggingface_hub[hf_xet]'")
            patterns = ["README.md", "index/*", "tokens/*", "tools/*"] + (["audio/*"] if args.with_audio else [])
            print("downloading {} ({}) ...".format(args.repo, ", ".join(patterns)), flush=True)
            root = Path(snapshot_download(args.repo, repo_type="dataset", allow_patterns=patterns, token=args.token or None,
                                          local_dir=args.cache_dir or None))
    missing = [p for p in ("index/conversations.parquet",) if not (root / p).exists()]
    if missing:
        sys.exit("incomplete copy under {}: missing {}".format(root, missing))
    return root


def ensure_file(root: Path, rel: str, args: argparse.Namespace) -> Path:
    """Return the local path of a repo file, downloading it on demand (audio tars when the local copy was made without them)."""
    p = root / rel
    if p.exists():
        return p
    from huggingface_hub import hf_hub_download
    print("  fetching {} ...".format(rel), flush=True)
    return Path(hf_hub_download(args.repo, rel, repo_type="dataset", token=args.token or None, local_dir=str(root)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output root; conversations go to <out>/Rosie_Stamp/<session_id>/<seg>/")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--local-dir", help="an existing `hf download ... --local-dir` copy of the repo")
    parser.add_argument("--cache-dir", help="where snapshot_download should put the repo files (default: the HF cache)")
    parser.add_argument("--token", help="HF token (default: HF_TOKEN env or the `hf auth login` token)")
    parser.add_argument("--with-audio", action="store_true", help="also unpack the FLAC lanes from audio/*.tar")
    parser.add_argument("--fetch", action="store_true", help="ignore the copy this script sits in and download from the Hub")
    parser.add_argument("--set", choices=["A", "B"], help="only conversations of one source set")
    parser.add_argument("--limit", type=int, help="at most N conversations (testing)")
    parser.add_argument("--force", action="store_true", help="rewrite files that already exist with the right size")
    args = parser.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is not installed: pip install pyarrow")
    try:
        import google_crc32c  # noqa: F401
        have_crc = True
    except ImportError:
        have_crc = False

    t0 = time.time()
    root = locate(args)
    idx_rows = pq.read_table(root / "index" / "conversations.parquet").to_pylist()
    idx_rows.sort(key=lambda r: (r["session_id"], r["seg"]))
    if args.set:
        idx_rows = [r for r in idx_rows if r["source_dataset"] == args.set]
    if args.limit:
        idx_rows = idx_rows[: args.limit]
    wanted = {(r["session_id"], r["seg"]) for r in idx_rows}
    hours = sum(r["clip_duration_sec"] for r in idx_rows) / 3600
    print("{}: {} conversations, {:.1f} h, {} with SAM streams -> {}".format(args.repo, len(idx_rows), hours, sum(1 for r in idx_rows if r["has_sam"]), args.out), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if (root / "README.md").exists():
        (args.out / "README.md").write_bytes((root / "README.md").read_bytes())

    shards = sorted({r["token_shard"] for r in idx_rows})
    stats = {"files_new": 0, "files_present": 0, "bytes": 0, "conversations": 0, "problems": []}
    manifest: List[dict] = []
    for shard in shards:
        path = ensure_file(root, shard, args)
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                key = (row["session_id"], row["seg"])
                if key not in wanted:
                    continue
                local = args.out / ACTOR_SLUG / row["session_id"] / row["seg"]
                files: Dict[str, str] = {}
                for col, name in TOKEN_COLS.items():
                    data = row.get(col)
                    if data is None:
                        continue
                    try:
                        shape, descr = npy_shape(data)
                        if shape != (32, row["mimi_T"]) or "i2" not in descr:
                            stats["problems"].append("{}/{} {}: shape {} dtype {} vs mimi_T {}".format(key[0], key[1], name, shape, descr, row["mimi_T"]))
                    except Exception as exc:  # noqa: BLE001
                        stats["problems"].append("{}/{} {}: {}".format(key[0], key[1], name, exc))
                    stats["files_" + write_bytes(local / name, data, args.force)] += 1
                    stats["bytes"] += len(data)
                    files[name] = col
                for col, name in JSON_COLS.items():
                    text = row.get(col)
                    if text is None:
                        continue
                    data = text.encode("utf-8")
                    stats["files_" + write_bytes(local / name, data, args.force)] += 1
                    stats["bytes"] += len(data)
                stats["conversations"] += 1
                cb = json.loads(row["crossbleed_json"])["summary"]
                manifest.append({"actor": "Rosie Stamp", "assistant_lane": row["rosie_lane"], "user_lane": row["user_lane"], "user_name": row.get("partner_name"),
                                 "session_id": row["session_id"], "seg": row["seg"], "source_dataset": row["source_dataset"], "recorded_date": row.get("recorded_date"),
                                 "clip_start_sec": row["clip_start_sec"], "clip_end_sec": row["clip_end_sec"], "clip_duration_sec": row["clip_duration_sec"],
                                 "mimi_T": row["mimi_T"], "has_sam": row["has_sam"],
                                 "assistant_crossbleed": cb["Rosie Stamp"]["status"], "user_crossbleed": cb["user"]["status"],
                                 "local_dir": str((local).relative_to(args.out)), "token_files": len(files)})
        print("  {}: {} conversations so far ({:.0f} MB written/checked)".format(shard, stats["conversations"], stats["bytes"] / 1e6), flush=True)

    audio_stats = None
    if args.with_audio:
        audio_stats = {"files_new": 0, "files_present": 0, "bytes": 0, "problems": [], "tars": 0}
        members_path = root / "index" / "audio_members.parquet"
        if not members_path.exists():
            try:
                members_path = ensure_file(root, "index/audio_members.parquet", args)
            except Exception as exc:  # noqa: BLE001
                print("audio: index/audio_members.parquet not in the repo yet (phase 2 still running?): {}".format(str(exc)[:200]), file=sys.stderr)
                members_path = None
        if members_path is not None:
            members = [m for m in pq.read_table(members_path).to_pylist() if (m["session_id"], m["seg"]) in wanted]
            by_tar: Dict[str, List[dict]] = {}
            for m in members:
                by_tar.setdefault(m["tar"], []).append(m)
            for tar_name in sorted(by_tar):
                todo = [m for m in by_tar[tar_name] if args.force or not ((args.out / m["member"]).exists() and (args.out / m["member"]).stat().st_size == m["size"])]
                audio_stats["files_present"] += len(by_tar[tar_name]) - len(todo)
                if not todo:
                    continue
                tpath = ensure_file(root, tar_name, args)
                audio_stats["tars"] += 1
                want = {m["member"]: m for m in todo}
                with tarfile.open(tpath, "r:") as tf:
                    for ti in tf:
                        m = want.get(ti.name)
                        if m is None or not ti.isreg():
                            continue
                        data = tf.extractfile(ti).read()
                        if len(data) != m["size"]:
                            audio_stats["problems"].append("{}: size {} != {}".format(ti.name, len(data), m["size"]))
                        elif have_crc:
                            import google_crc32c
                            if base64.b64encode(google_crc32c.Checksum(data).digest()).decode("ascii") != m["crc32c"]:
                                audio_stats["problems"].append("{}: crc32c mismatch".format(ti.name))
                        audio_stats["files_" + write_bytes(args.out / m["member"], data, True)] += 1
                        audio_stats["bytes"] += len(data)
                print("  {}: {} FLACs ({:.0f} MB)".format(tar_name, len(todo), sum(m["size"] for m in todo) / 1e6), flush=True)

    if manifest:
        with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
            w.writeheader()
            w.writerows(manifest)
    (args.out / "_manifests").mkdir(exist_ok=True)
    summary = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "repo": args.repo, "source_copy": str(root),
               "conversations": stats["conversations"], "conversation_hours": round(hours, 3), "tokens_and_labels": stats, "audio": audio_stats,
               "elapsed_s": round(time.time() - t0, 1)}
    (args.out / "_manifests" / "download_summary.{}.json".format(time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    problems = stats["problems"] + (audio_stats["problems"] if audio_stats else [])
    print("done: {} conversations ({:.1f} h), {} files written + {} already present, {:.2f} GB{}; {} problem(s); outputs under {}".format(
        stats["conversations"], hours, stats["files_new"] + (audio_stats["files_new"] if audio_stats else 0),
        stats["files_present"] + (audio_stats["files_present"] if audio_stats else 0), (stats["bytes"] + (audio_stats["bytes"] if audio_stats else 0)) / 1e9,
        " (incl. {} FLACs)".format(audio_stats["files_new"] + audio_stats["files_present"]) if audio_stats else "", len(problems), args.out))
    for p in problems[:20]:
        print("   ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
