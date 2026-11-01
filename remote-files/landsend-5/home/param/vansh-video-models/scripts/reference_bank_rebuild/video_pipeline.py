#!/usr/bin/env python3
"""Resumable stage-2 reference-bank pipeline, using local Hugging Face sources.

Commands in order:
  targets --upstream upstream-selection.parquet --video-index hf-video-index.json
  # Download output-dir/downloads.tsv with hf_sources.py download.
  scan --workers 32
  # Or run scan --shard N --num-shards K on each node, then:
  # aggregate-scan --num-shards K
  select --upstream upstream-selection.parquet
  decode --workers 8

Every command needs --output-dir; targets also needs --source-root. Missing HF
sources are recorded and block a complete run. --allow-incomplete explicitly
permits processing available targets while preserving incomplete status.

Dedup follows the archived generation implementation: the 10-second check
includes upstream references, and thumbnail MAD compares accepted scan frames.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

# Set before NumPy/pandas imports and before spawned measurement workers import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd

from hf_sources import write_json
from measurements import ARCHIVE, MODELS, initialize_measurements
from upstream import BLACKBANDS_CSV, CSV_ROOT, load_corpus
from video_scan import (build_scan_download_plan, choose_scan_tracks,
                        decode_selected_crops, scan_chunk, select_scanned, validate_video)

MODULES = None
WORKER_CONFIG = None


def read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_table(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        table.to_parquet(temporary, index=False)
    else:
        table.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def task_path(output_dir, track, chunk):
    return Path(output_dir) / "candidates" / track / f"{int(chunk):06d}.parquet"


def input_fingerprint(track, chunk, pose_path, config):
    """Bind resume to exact poses, source identity, policies, code, and video stat.

    Full videos are content-verified by hf_sources before scanning. Rehashing a
    multi-GB video per chunk would add redundant reads; pinned HF Xet metadata
    plus local size/mtime/inode detect local replacements for checkpoint reuse.
    """
    stat = Path(track["video_path"]).stat()
    payload = {"track": track, "chunk": int(chunk), "pose_sha256": digest(pose_path),
               "video_stat": [stat.st_size, stat.st_mtime_ns, stat.st_ino], "code": config}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def code_config(archive_root, model_root):
    # Logical names let an immutable shared snapshot and the checkout validate
    # each other. Their locations do not alter measurements or fingerprints.
    paths = {f"rebuild/{name}": Path(__file__).with_name(name)
             for name in ("video_pipeline.py", "video_scan.py", "measurements.py")}
    paths.update({f"archive/{name}": Path(archive_root) / name for name in (
        "node/ref_audit/audit.py", "node/ref_audit/audit59.py",
        "node/mouth_open_pilot/teeth.py", "node/mouth_open_pilot/allteeth.py",
        "node/mouth_open_pilot/scan.py")})
    paths.update({f"models/{name}": Path(model_root) / name
                  for name in ("face_landmarker.task", "hand_landmarker.task")})
    return {name: digest(path) for name, path in paths.items()}


def shard_tracks(tracks, num_shards):
    """Assign whole tracks once, balancing the number of scanned chunks."""
    if len({row["track"] for row in tracks}) != len(tracks):
        raise ValueError("Duplicate tracks in scan plan")
    if any(len(set(row["usable_chunks"])) != len(row["usable_chunks"]) for row in tracks):
        raise ValueError("Duplicate chunks in scan plan")
    groups, loads = [[] for _ in range(num_shards)], [0] * num_shards
    for track in sorted(tracks, key=lambda row: (-len(row["usable_chunks"]), row["track"])):
        shard = min(range(num_shards), key=lambda i: (loads[i], i))
        groups[shard].append(track)
        loads[shard] += len(track["usable_chunks"])
    return [sorted(group, key=lambda row: row["track"]) for group in groups]


def shard_report_dir(output_dir, shard, num_shards):
    return Path(output_dir) / "scan-shards" / f"shard-{shard:05d}-of-{num_shards:05d}"


def plan_identity(output_dir):
    return {name: digest(Path(output_dir) / name) for name in ("tracks.json", "targets-summary.json")}


def shard_identity(args, tracks):
    return {"shard": args.shard, "num_shards": args.num_shards,
            "plan_sha256": plan_identity(args.output_dir),
            "assigned_tracks": [row["track"] for row in tracks],
            "assigned_chunks": sum(len(row["usable_chunks"]) for row in tracks)}


def resume_ok(path, fingerprint):
    receipt_path = path.with_suffix(".complete.json")
    if not path.is_file() or not receipt_path.is_file():
        return False
    receipt = json.loads(receipt_path.read_text())
    return receipt.get("fingerprint") == fingerprint and receipt.get("sha256") == digest(path)


def initialize_worker(archive_root, model_root, config):
    global MODULES, WORKER_CONFIG
    MODULES = initialize_measurements(archive_root, model_root)
    WORKER_CONFIG = config


def scan_one(task):
    track, chunk, path = task
    pose_path = track["pose_paths"].get(str(chunk), track["pose_paths"].get(chunk))
    fingerprint = input_fingerprint(track, chunk, pose_path, WORKER_CONFIG)
    rows = scan_chunk(track, chunk, pose_path, MODULES)
    atomic_table(rows, path)
    receipt = {"track": track["track"], "chunk": chunk, "rows": len(rows),
               "fingerprint": fingerprint, "sha256": digest(path), "complete": True}
    write_json(Path(path).with_suffix(".complete.json"), receipt)
    return receipt


def load_plan(args):
    report = json.loads((args.output_dir / "targets-summary.json").read_text())
    if not report["complete"] and not args.allow_incomplete:
        raise ValueError("HF target coverage is incomplete; inspect targets-missing.json. "
                         "Use --allow-incomplete only to process the available subset.")
    return json.loads((args.output_dir / "tracks.json").read_text()), report


def targets(args):
    corpus = load_corpus(args.csv_root, args.blackbands_csv)
    upstream = read_table(args.upstream)
    ranked = choose_scan_tracks(corpus.tracks, upstream, max_tracks=None)
    video_index = json.loads(args.video_index.read_text())
    tracks, downloads, unavailable, attempted = [], [], [], []
    selected_counts = defaultdict(int)
    actor_key = lambda row: (row.get("bank", "bank"), row["uuid"])
    for row in ranked:
        actor = actor_key(row)
        if selected_counts[actor] == 2:
            continue
        attempted.append(row)
        try:
            planned, pairs = build_scan_download_plan([row], video_index, args.source_root, corpus)
            tracks.extend(planned)
            downloads.extend(pairs)
            selected_counts[actor] += 1
        except ValueError as exc:
            # The archive continues down the ranking when a source cannot be
            # resolved. Keep the failed attempt visible; never pick an
            # ambiguous HF video or treat it as a successful scan.
            unavailable.append({**row, "error": str(exc)})
    # Unavailable alternatives are diagnostics, not missing selected targets.
    # Keep conservative source-limited status only where such alternatives
    # remain unresolved after exhausting the ranking with fewer than two picks.
    missing = [row for row in unavailable if selected_counts[actor_key(row)] < 2]
    selected_by_track = {(actor_key(row), row["track"]): row for row in tracks}
    unavailable_by_track = {(actor_key(row), row["track"]): row for row in unavailable}
    actor_evidence = {}
    for row in ranked:
        actor = actor_key(row)
        evidence = actor_evidence.setdefault(actor, {
            "bank": actor[0], "uuid": actor[1], "selected_tracks": [],
            "unavailable_tracks": [], "ranked_tracks": []})
        entry = {"rank": len(evidence["ranked_tracks"]) + 1, "track": row["track"],
                 "usable_chunks": row["usable_chunks"]}
        key = (actor, row["track"])
        if key in selected_by_track:
            entry.update(status="selected", source=selected_by_track[key]["source"])
            evidence["selected_tracks"].append(row["track"])
        elif key in unavailable_by_track:
            entry.update(status="unavailable", error=unavailable_by_track[key]["error"])
            evidence["unavailable_tracks"].append(row["track"])
        else:
            entry["status"] = "not-examined-two-sources-selected"
        evidence["ranked_tracks"].append(entry)
    for evidence in actor_evidence.values():
        evidence["selected_track_count"] = len(evidence["selected_tracks"])
        evidence["ranking_exhausted"] = all(row["status"] != "not-examined-two-sources-selected"
                                             for row in evidence["ranked_tracks"])
        evidence["source_limited"] = evidence["selected_track_count"] < 2 and bool(evidence["unavailable_tracks"])
    write_json(args.output_dir / "targets-all.json", attempted)
    write_json(args.output_dir / "targets-missing.json", missing)
    write_json(args.output_dir / "targets-unavailable.json", unavailable)
    write_json(args.output_dir / "targets-actors.json", list(actor_evidence.values()))
    write_json(args.output_dir / "tracks.json", tracks)
    (args.output_dir / "downloads.tsv").write_text("".join(f"{key}\t{path}\n" for key, path in downloads))
    summary = {"upstream_sha256": digest(args.upstream), "video_index_sha256": digest(args.video_index),
               "corpus_source_hashes": corpus.source_hashes, "requested_tracks": len(attempted),
               "requested_actors": len(selected_counts),
               "available_tracks": len(tracks), "available_actors": len({row["uuid"] for row in tracks}),
               "missing_tracks": len(missing), "missing_actors": len({row["uuid"] for row in missing}),
               "unavailable_tracks": len(unavailable),
               "source_limited_actors": sum(row["source_limited"] for row in actor_evidence.values()),
               "actors_by_available_track_count": {str(n): sum(count == n for count in selected_counts.values())
                                                   for n in range(3)},
               "track_selection_policy": "rank-by-usable-chunks-then-track; resolve-until-two-available",
               "pose_files": sum(len(row["pose_paths"]) for row in tracks),
               "video_bytes": sum(row["video_source_metadata"]["size"] for row in tracks),
               "download_files": len(downloads), "complete": not missing}
    write_json(args.output_dir / "targets-summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["complete"] or args.allow_incomplete else 1


def scan(args):
    tracks, target_report = load_plan(args)
    tracks = shard_tracks(tracks, args.num_shards)[args.shard]
    report_dir = (args.output_dir if args.num_shards == 1 else
                  shard_report_dir(args.output_dir, args.shard, args.num_shards))
    report_dir.mkdir(parents=True, exist_ok=True)
    # Hold this descriptor for the whole scan. Concurrent resume attempts must
    # not overwrite candidates or advertise completion while work is active.
    with (report_dir / "scan.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return scan_locked(args, tracks, target_report, report_dir)


def scan_locked(args, tracks, target_report, report_dir):
    config = code_config(args.archive_root, args.model_root)
    identity = shard_identity(args, tracks)
    # A failed restart cannot leave an earlier successful summary in place.
    (report_dir / "scan-summary.json").unlink(missing_ok=True)
    write_json(report_dir / "scan-code-inputs.json", config)
    pending, completed, missing = [], [], []
    for track_number, track in enumerate(tracks, 1):
        video = Path(track["video_path"])
        if not video.is_file():
            missing.append({"track": track["track"], "path": str(video), "type": "video"})
            continue
        validate_video(video)
        for chunk in track["usable_chunks"]:
            pose_path = track["pose_paths"].get(str(chunk), track["pose_paths"].get(chunk))
            if not Path(pose_path).is_file():
                missing.append({"track": track["track"], "path": pose_path, "type": "poses", "chunk": chunk})
                continue
            path = task_path(args.output_dir, track["track"], chunk)
            fingerprint = input_fingerprint(track, chunk, pose_path, config)
            if resume_ok(path, fingerprint):
                completed.append(json.loads(path.with_suffix(".complete.json").read_text()))
            else:
                pending.append((track, chunk, path))
        if track_number % 50 == 0:
            print(json.dumps({"preflight_tracks": track_number, "total_tracks": len(tracks),
                              "missing_sources": len(missing), "pending_chunks": len(pending)}), flush=True)
    write_json(report_dir / "scan-missing.json", missing)
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(f"{len(missing)} local HF sources missing; inspect {report_dir / 'scan-missing.json'}")
    total_pending = len(pending)
    started = time.monotonic()
    print(json.dumps({"pending_chunks": total_pending, "resumed_chunks": len(completed), "workers": args.workers}), flush=True)
    failures = []
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker,
                                 initargs=(args.archive_root, args.model_root, config)) as pool:
            futures = {pool.submit(scan_one, task): (task[0]["track"], task[1]) for task in pending}
            for count, future in enumerate(as_completed(futures), 1):
                try:
                    completed.append(future.result())
                except Exception as exc:
                    track, chunk = futures[future]
                    failures.append({"track": track, "chunk": chunk, "error": f"{type(exc).__name__}: {exc}"})
                if count % 10 == 0 or count == total_pending:
                    print(json.dumps({"finished": count, "pending_total": total_pending, "failures": len(failures),
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    write_json(report_dir / "scan-failures.json", failures)
    summary = {"completed_chunks": len(completed), "candidate_rows": sum(row["rows"] for row in completed),
               "missing_sources": len(missing), "failures": len(failures),
               "available_targets_complete": not missing and not failures,
               "complete": target_report["complete"] and not missing and not failures,
               **identity}
    write_json(report_dir / "scan-summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["complete"] or (args.allow_incomplete and not failures) else 1


def aggregate_scan(args):
    """Validate every shard and checkpoint before writing global reports."""
    tracks, target_report = load_plan(args)
    config = code_config(args.archive_root, args.model_root)
    groups = shard_tracks(tracks, args.num_shards)
    completed, missing, failures = [], [], []
    plan = plan_identity(args.output_dir)
    for shard, group in enumerate(groups):
        report_dir = shard_report_dir(args.output_dir, shard, args.num_shards)
        if args.num_shards == 1:
            report_dir = args.output_dir
        report = json.loads((report_dir / "scan-summary.json").read_text())
        expected = {"shard": shard, "num_shards": args.num_shards, "plan_sha256": plan,
                    "assigned_tracks": [row["track"] for row in group],
                    "assigned_chunks": sum(len(row["usable_chunks"]) for row in group)}
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Stale or mismatched scan shard: {report_dir}")
        if json.loads((report_dir / "scan-code-inputs.json").read_text()) != config:
            raise ValueError(f"Scan code/model changed: {report_dir}")
        shard_missing = json.loads((report_dir / "scan-missing.json").read_text())
        shard_failures = json.loads((report_dir / "scan-failures.json").read_text())
        absent = {(row["track"], row.get("chunk")) for row in shard_missing + shard_failures}
        receipts = []
        for track in group:
            for chunk in track["usable_chunks"]:
                if (track["track"], None) in absent or (track["track"], chunk) in absent:
                    continue
                path = task_path(args.output_dir, track["track"], chunk)
                pose = track["pose_paths"].get(str(chunk), track["pose_paths"].get(chunk))
                fingerprint = input_fingerprint(track, chunk, pose, config)
                if not resume_ok(path, fingerprint):
                    raise ValueError(f"Missing or changed scan checkpoint: {path}")
                receipt = json.loads(path.with_suffix(".complete.json").read_text())
                if receipt.get("track") != track["track"] or receipt.get("chunk") != chunk:
                    raise ValueError(f"Scan checkpoint identity mismatch: {path}")
                receipts.append(receipt)
        if (report["completed_chunks"] != len(receipts) or
                report["candidate_rows"] != sum(row["rows"] for row in receipts) or
                report["missing_sources"] != len(shard_missing) or report["failures"] != len(shard_failures)):
            raise ValueError(f"Scan shard totals disagree with verified checkpoints: {report_dir}")
        completed.extend(receipts)
        missing.extend(shard_missing)
        failures.extend(shard_failures)
    write_json(args.output_dir / "scan-code-inputs.json", config)
    write_json(args.output_dir / "scan-missing.json", missing)
    write_json(args.output_dir / "scan-failures.json", failures)
    summary = {"completed_chunks": len(completed), "candidate_rows": sum(row["rows"] for row in completed),
               "missing_sources": len(missing), "failures": len(failures), "num_shards": args.num_shards,
               "plan_sha256": plan, "available_targets_complete": not missing and not failures,
               "complete": target_report["complete"] and not missing and not failures}
    if args.num_shards == 1:
        summary.update(shard=0, assigned_tracks=[row["track"] for row in groups[0]],
                       assigned_chunks=sum(len(row["usable_chunks"]) for row in groups[0]))
    write_json(args.output_dir / "scan-summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["complete"] or (args.allow_incomplete and not failures) else 1


def select(args):
    tracks, target_report = load_plan(args)
    if digest(args.upstream) != target_report["upstream_sha256"]:
        raise ValueError("Upstream selection changed since targets were planned; rerun targets")
    upstream = read_table(args.upstream)
    config = json.loads((args.output_dir / "scan-code-inputs.json").read_text())
    if config != code_config(args.archive_root, args.model_root):
        raise ValueError("Scan implementation/model changed; rerun scan before selection")
    scan_report = json.loads((args.output_dir / "scan-summary.json").read_text())
    if scan_report.get("plan_sha256") != plan_identity(args.output_dir):
        raise ValueError("Scan plan changed or distributed shards were not aggregated; rerun scan")
    grouped = defaultdict(list)
    for track in tracks:
        grouped[(track.get("bank", "bank"), track["uuid"])].append(track)
    output, missing = [], []
    for (bank, uuid), actor_tracks in sorted(grouped.items()):
        chunks = []
        for track in actor_tracks:
            for chunk in track["usable_chunks"]:
                path = task_path(args.output_dir, track["track"], chunk)
                receipt = path.with_suffix(".complete.json")
                if not path.is_file() or not receipt.is_file():
                    missing.append({"track": track["track"], "chunk": chunk})
                    continue
                meta = json.loads(receipt.read_text())
                pose_path = track["pose_paths"].get(str(chunk), track["pose_paths"].get(chunk))
                fingerprint = input_fingerprint(track, chunk, pose_path, config)
                if meta["fingerprint"] != fingerprint:
                    raise ValueError(f"Scan sources/policy changed; rerun scan: {path}")
                if meta["sha256"] != digest(path):
                    raise ValueError(f"Candidate file changed after scan: {path}")
                rows = pd.read_parquet(path)
                if len(rows):
                    chunks.append(rows)
        if chunks:
            old = upstream[upstream.uuid == uuid]
            if "bank" in old:
                old = old[old.bank == bank]
            rows = select_scanned(pd.concat(chunks, ignore_index=True), old, args.fit_score)
            if len(rows):
                output.append(rows)
    write_json(args.output_dir / "selection-missing.json", missing)
    if missing and not args.allow_incomplete:
        raise ValueError(f"{len(missing)} scan chunks incomplete; inspect selection-missing.json")
    selected = pd.concat(output, ignore_index=True) if output else pd.DataFrame(columns=["uuid", "property", "index", "track", "frame_index"])
    selected = selected.sort_values(["uuid", "property", "index"], kind="stable").reset_index(drop=True)
    atomic_table(selected, args.output_dir / "video-selected.parquet")
    atomic_table(selected, args.output_dir / "video-selected.csv")
    summary = {"rows": len(selected), "actors": int(selected.uuid.nunique()),
               "missing_chunks": len(missing), "complete": target_report["complete"] and not missing,
               "slot_counts": {str(k): int(v) for k, v in selected.groupby("property").size().items()},
               "fit_score_sha256": digest(args.fit_score),
               "dedup_note": "10s same-track checks include upstream selections; actor-wide MAD<6 checks include upstream and selected scan images; P7-9 share a group and require 2s within a recording"}
    write_json(args.output_dir / "selection-summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["complete"] or args.allow_incomplete else 1


def decode_one(track, selected, output_dir):
    receipt_path = output_dir / "decoded" / f"{track['track'].replace('/', '__')}.json"
    stat = Path(track["video_path"]).stat()
    context = {"rows": selected.to_json(orient="records", double_precision=15), "track": track,
               "video_stat": [stat.st_size, stat.st_mtime_ns, stat.st_ino],
               "decoder_code": digest(Path(__file__).with_name("video_scan.py"))}
    row_hash = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt["selected_sha256"] == row_hash and all(Path(row["image_path"]).is_file() and
                digest(row["image_path"]) == row["image_sha256"] for row in receipt["images"]):
            result = selected.copy()
            result["image_path"] = [row["image_path"] for row in receipt["images"]]
            result["prepared_image"] = False
            return result
    result = decode_selected_crops(selected, {track["track"]: track}, output_dir / "video-frames")
    receipt = {"selected_sha256": row_hash,
               "images": [{"image_path": path, "image_sha256": digest(path)} for path in result.image_path]}
    write_json(receipt_path, receipt)
    return result


def decode(args):
    tracks, target_report = load_plan(args)
    selection_report = json.loads((args.output_dir / "selection-summary.json").read_text())
    if not selection_report["complete"] and not args.allow_incomplete:
        raise ValueError("Scan/selection is incomplete; inspect selection-summary.json")
    selected = pd.read_parquet(args.output_dir / "video-selected.parquet")
    lookup = {row["track"]: row for row in tracks}
    outputs, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(decode_one, lookup[track], rows, args.output_dir): track
                   for track, rows in selected.groupby("track", sort=True)}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                outputs.append(future.result())
            except Exception as exc:
                failures.append({"track": futures[future], "error": f"{type(exc).__name__}: {exc}"})
            if count % 50 == 0 or count == len(futures):
                print(json.dumps({"decoded_tracks": count, "total_tracks": len(futures), "failures": len(failures)}), flush=True)
    write_json(args.output_dir / "decode-failures.json", failures)
    if failures:
        raise ValueError(f"{len(failures)} tracks failed to decode; inspect decode-failures.json")
    decoded = pd.concat(outputs, ignore_index=True) if outputs else selected.assign(image_path="", prepared_image=False)
    decoded = decoded.sort_values(["uuid", "property", "index"], kind="stable").reset_index(drop=True)
    atomic_table(decoded, args.output_dir / "video-encoding-input.parquet")
    atomic_table(decoded, args.output_dir / "video-encoding-input.csv")
    summary = {"rows": len(decoded), "complete": target_report["complete"] and selection_report["complete"],
               "full_rgb_dimensions": [512, 960], "prepared_image": False}
    write_json(args.output_dir / "decode-summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["complete"] or args.allow_incomplete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("targets", "scan", "aggregate-scan", "select", "decode"):
        command = commands.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--allow-incomplete", action="store_true")
        if name in ("targets", "select"):
            command.add_argument("--upstream", type=Path, required=True)
        if name == "targets":
            command.add_argument("--source-root", type=Path, required=True)
            command.add_argument("--video-index", type=Path, required=True)
            command.add_argument("--csv-root", type=Path, default=CSV_ROOT)
            command.add_argument("--blackbands-csv", type=Path, default=BLACKBANDS_CSV)
        if name in ("scan", "aggregate-scan", "select"):
            command.add_argument("--archive-root", type=Path, default=ARCHIVE)
            command.add_argument("--model-root", type=Path, default=MODELS)
        if name in ("scan", "aggregate-scan"):
            command.add_argument("--num-shards", type=int, default=1)
        if name == "scan":
            command.add_argument("--shard", type=int, default=0)
        if name == "select":
            command.add_argument("--fit-score", type=Path, default=ARCHIVE / "fit_score.parquet")
        if name in ("scan", "decode"):
            command.add_argument("--workers", type=int, default=32 if name == "scan" else 8)
    args = parser.parse_args()
    if hasattr(args, "workers") and args.workers < 1:
        parser.error("--workers must be positive")
    if hasattr(args, "num_shards") and args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if hasattr(args, "shard") and not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, --num-shards)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return globals()[args.command.replace("-", "_")](args)


if __name__ == "__main__":
    raise SystemExit(main())
