#!/usr/bin/env python3
"""Attach authoritative stage2 black-band metadata without rewriting training data.

Build a checksummed release, then install it on a node only after checking the
original row metadata. All output is additive; training manifests remain valid.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = "stage2-black-band-annotations-v1"
REPO = "InternalCan/stage2-handover-pose-mimi-sam-v1"
LANES = ("speaker_00", "speaker_01")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def interval_ticks(intervals, T):
    """Recover exact 25 Hz ticks from the source index's float32 timestamps."""
    if T < 0 or int(T) != T:
        raise ValueError(f"Invalid T: {T}")
    out = []
    for pair in intervals if intervals is not None else []:
        if len(pair) != 2:
            raise ValueError(f"Invalid interval: {pair}")
        a, b = map(float, pair)
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError(f"Non-finite interval: {pair}")
        x, y = round(a * 25), round(b * 25)
        if abs(a * 25 - x) > .01 or abs(b * 25 - y) > .01:
            raise ValueError(f"Interval is not on the 25 Hz grid: {pair}")
        if not 0 <= x < y <= 2 * T:
            raise ValueError(f"Interval outside T={T}: {pair}")
        if out and x < out[-1][1]:
            raise ValueError(f"Overlapping or unsorted intervals: {pair}")
        out.append((x, y))
    return out


def lane_annotation(intervals, T, start_token=0, n_tokens=None,
                    min_ticks=25, available=True):
    n_tokens = T - start_token if n_tokens is None else n_tokens
    if not 0 <= start_token <= start_token + n_tokens <= T:
        raise ValueError(f"Invalid crop: {start_token}, {n_tokens}, T={T}")
    if min_ticks < 1:
        raise ValueError("min_ticks must be positive")
    ticks = interval_ticks(intervals, T)
    if not available and ticks:
        raise ValueError("Unavailable lane contains intervals")
    left, right = 2 * start_token, 2 * (start_token + n_tokens)
    clipped = [(max(a, left) - left, min(b, right) - left)
               for a, b in ticks if b - a >= min_ticks and a < right and b > left]
    return {
        "has_black_band": bool(clipped) if available else None,
        "annotation_available": bool(available),
        "intervals_seconds": [[a / 25, b / 25] for a, b in clipped],
        "pose_tick_intervals": [list(iv) for iv in clipped],
        "token_intervals": [[a // 2, (b + 1) // 2] for a, b in clipped],
        "n": len(clipped),
        "seconds": sum(b - a for a, b in clipped) / 25,
    }


def combine_flags(flags):
    flags = list(flags)
    return True if any(x is True for x in flags) else (None if any(x is None for x in flags) else False)


def annotate(source, T, start=0, n=None):
    lanes = {}
    for lane in LANES:
        iv = None if source is None else source[lane + "_no_detection_intervals"]
        lanes[lane] = lane_annotation(iv, T, start, n, available=iv is not None)
    flag = combine_flags(v["has_black_band"] for v in lanes.values())
    complete = all(v["annotation_available"] for v in lanes.values())
    status = ("source_conversation_missing" if source is None else
              "band_detected" if flag is True else
              "no_band_detected" if flag is False else "pose_annotation_unavailable")
    result = {"has_black_band": flag, "black_band_annotation_complete": complete,
              "black_band_status": status, "black_band_min_seconds": 1.0}
    for lane, info in lanes.items():
        result.update({lane + "_black_band_" + k: v for k, v in info.items()})
    return result


def append_annotations(table, values):
    """Use explicit list types, including when every interval list is empty."""
    if len(table) != len(values):
        raise ValueError("Annotation row count differs")
    if not values:
        raise ValueError("Empty input table")
    for key in values[0]:
        if key in table.column_names:
            raise ValueError(f"Refusing to replace existing column: {key}")
        if key.endswith("intervals_seconds"):
            dtype = pa.list_(pa.list_(pa.float64()))
        elif key.endswith(("token_intervals", "pose_tick_intervals")):
            dtype = pa.list_(pa.list_(pa.int64()))
        elif key.endswith(("has_black_band", "annotation_available", "annotation_complete")):
            dtype = pa.bool_()
        elif key.endswith("_n"):
            dtype = pa.int64()
        elif key.endswith("seconds"):
            dtype = pa.float64()
        else:
            dtype = None
        table = table.append_column(key, pa.array([v[key] for v in values], type=dtype))
    meta = dict(table.schema.metadata or {})
    meta[b"black_band_schema"] = SCHEMA.encode()
    meta[b"black_band_clocks"] = b"half-open; row-relative seconds, 25Hz pose ticks, 12.5Hz tokens"
    return table.replace_schema_metadata(meta)


def counts(values):
    return {"rows": len(values), "banded": sum(v["has_black_band"] is True for v in values),
            "not_banded": sum(v["has_black_band"] is False for v in values),
            "unknown": sum(v["has_black_band"] is None for v in values),
            "incomplete_annotations": sum(not v["black_band_annotation_complete"] for v in values),
            "status_counts": dict(Counter(v["black_band_status"] for v in values))}


def build(source_root, dataset_root, output, source_revision):
    source_root, dataset_root, output = map(Path, (source_root, dataset_root, output))
    output.mkdir(parents=True, exist_ok=False)
    idx_path = source_root / "tools/shard_index.parquet"
    scalar_path = source_root / "index/conversations.parquet"
    idx = pq.read_table(idx_path)
    rows = idx.to_pylist()
    source = {(r["session_id"], r["seg"]): r for r in rows}
    if len(source) != len(rows):
        raise ValueError("Duplicate source conversation")
    scalar = pq.read_table(scalar_path)
    if {(r["session_id"], r["seg"]) for r in scalar.select(["session_id", "seg"]).to_pylist()} != set(source):
        raise ValueError("Source indexes have different membership")
    shutil.copy2(idx_path, output / "source_shard_index.parquet")
    shutil.copy2(scalar_path, output / "source_conversations.parquet")
    shutil.copy2(__file__, output / "annotate.py")
    values = [annotate(r, r["T"]) for r in rows]
    full = append_annotations(idx, values)
    # Exactly the supplied script's null-as-empty flag; has_black_band is stricter.
    full = full.append_column("banded", pa.array([v["has_black_band"] is True for v in values], type=pa.bool_()))
    pq.write_table(full, output / "conversations.parquet", compression="zstd")
    pq.write_table(full.filter(full["banded"]), output / "banded_conversations.parquet", compression="zstd")
    all_stats = {"source": counts(values)}
    all_stats["source"]["conversations_with_any_no_detection_interval"] = sum(
        any(r[lane + "_no_detection_intervals"] for lane in LANES) for r in rows)
    all_stats["source"]["qualifying_speaker_intervals"] = sum(
        v[lane + "_black_band_n"] for v in values for lane in LANES)
    inputs = {}
    attachments = {}
    missing = []
    by_store = {}
    input_paths = sorted(dataset_root.glob("original/stores/*/rows.parquet")) + sorted(dataset_root.glob("s2h-m1-samsep/stores/*/rows.parquet"))
    if not input_paths:
        raise ValueError("No local stage2 stores")
    for path in input_paths:
        relative = path.relative_to(dataset_root).as_posix()
        inputs[relative] = sha256(path)
        table = pq.read_table(path)
        annotations = []
        for r in table.to_pylist():
            src = source.get(tuple(r["src_conversation"].split("/")))
            if src is not None and src["T"] != r["src_frames"]:
                raise ValueError(f"Source length mismatch: {r['row_key']}")
            if r["frame_end"] != r["frame_start"] + r["n_frames"]:
                raise ValueError(f"Row length mismatch: {r['row_key']}")
            if abs(r["chunk_start_sec"] * 12.5 - r["frame_start"]) > 1e-5:
                raise ValueError(f"Row clock mismatch: {r['row_key']}")
            ann = annotate(src, r["src_frames"], r["frame_start"], r["n_frames"])
            ann["black_band_source_shard"] = src["shard"] if src else None
            ann["black_band_source_row_index"] = src["row_index"] if src else None
            for role in ("assistant", "user"):
                lane = r[role + "_speaker_dir"].lower()
                if lane not in LANES:
                    raise ValueError(f"Unknown speaker: {lane}")
                ann[role + "_has_black_band"] = ann[lane + "_black_band_has_black_band"]
                for name in ("intervals_seconds", "token_intervals", "pose_tick_intervals"):
                    ann[role + "_black_band_" + name] = ann[lane + "_black_band_" + name]
            annotations.append(ann)
            if src is None:
                missing.append({"store": str(path.parent.relative_to(dataset_root)), "row": r["row"],
                                "row_key": r["row_key"], "src_conversation": r["src_conversation"],
                                "reason": "Conversation absent from supplied v1 source index"})
        out_relative = "rows/" + relative
        target = output / out_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(append_annotations(table, annotations), target, compression="zstd")
        all_stats[relative] = counts(annotations)
        attachments[str(path.with_name("rows.black_bands.parquet").relative_to(dataset_root))] = out_relative
        by_store[str(path.parent)] = dict(zip(table["row"].to_pylist(), annotations))
    pose_path = dataset_root / "pose/POSE_INDEX.json"
    inputs["pose/POSE_INDEX.json"] = sha256(pose_path)
    pose = json.loads(pose_path.read_text())
    pose_rows, pose_values = [], []
    for store_i, rows_by_id in pose["rows"].items():
        store = pose["stores"][int(store_i)]
        for row_id, row in rows_by_id.items():
            ann = by_store[store][int(row_id)]
            pose_rows.append({"store_index": int(store_i), "row": int(row_id), "row_key": row["row_key"],
                              "src_conversation": row["conv"], "n_frames": row["n"], "T": row["T"], "split": row["split"]})
            pose_values.append(ann)
    pq.write_table(append_annotations(pa.Table.from_pylist(pose_rows), pose_values), output / "pose_rows.parquet", compression="zstd")
    attachments["pose/rows.black_bands.parquet"] = "pose_rows.parquet"
    all_stats["pose_index"] = counts(pose_values)
    write_json(output / "unmatched_rows.json", missing)
    # Persist immutable provenance and enough information to re-check the join on every node.
    manifest = {"schema": SCHEMA, "source_repo": REPO, "source_revision": source_revision,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_index_sha256": sha256(idx_path), "source_conversations_sha256": sha256(scalar_path),
                "script_sha256": sha256(__file__), "band_min_seconds": 1.0, "confidence_threshold": 0.3,
                "inputs": inputs, "attachments": attachments, "summary": all_stats,
                "unknown_conversations": len({r["src_conversation"] for r in missing}),
                "excluded_dataset": "xabi-v2pure-cache: separate release; no source-ID crosswalk or raw confidence poses"}
    (output / "README.md").write_text(README)
    manifest["files"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    release_id = hashlib.sha256(json.dumps({"files": manifest["files"], "inputs": inputs}, sort_keys=True).encode()).hexdigest()[:20]
    manifest["release_id"] = release_id
    write_json(output / "MANIFEST.json", manifest)
    return manifest


def check_files(root, files):
    for rel, digest in files.items():
        if sha256(Path(root) / rel) != digest:
            raise ValueError(f"Hash mismatch: {Path(root) / rel}")


def install(package, dataset_root):
    package, dataset_root = map(Path, (package, dataset_root))
    manifest = json.loads((package / "MANIFEST.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unknown package schema")
    check_files(package, manifest["files"])
    check_files(dataset_root, manifest["inputs"])
    release = dataset_root / "black_bands/releases" / manifest["release_id"]
    release.parent.mkdir(parents=True, exist_ok=True)
    if release.exists():
        check_files(release, manifest["files"])
        if sha256(release / "MANIFEST.json") != sha256(package / "MANIFEST.json"):
            raise ValueError("Existing release manifest differs")
    else:
        tmp = release.with_name(release.name + f".staging-{os.getpid()}")
        shutil.copytree(package, tmp)
        check_files(tmp, manifest["files"])
        tmp.rename(release)
    for rel, target in manifest["attachments"].items():
        dest = dataset_root / rel
        expected = release / target
        if dest.exists() or dest.is_symlink():
            if not dest.is_symlink() or dest.resolve() != expected.resolve():
                raise ValueError(f"Refusing to replace unrelated attachment: {dest}")
        else:
            dest.symlink_to(os.path.relpath(expected, dest.parent))
    marker = dataset_root / "BLACK_BANDS.json"
    if marker.exists() and json.loads(marker.read_text()).get("schema") != SCHEMA:
        raise ValueError(f"Unrelated existing marker: {marker}")
    write_json(marker, {"schema": SCHEMA, "release_id": manifest["release_id"],
                        "manifest": str(release / "MANIFEST.json"),
                        "conversations": str(release / "conversations.parquet"),
                        "banded_conversations": str(release / "banded_conversations.parquet"),
                        "readme": str(release / "README.md"),
                        "row_filename": "rows.black_bands.parquet",
                        "source_repo": REPO, "band_min_seconds": 1.0})
    return verify(dataset_root)


def verify(dataset_root):
    root = Path(dataset_root)
    marker = json.loads((root / "BLACK_BANDS.json").read_text())
    release = Path(marker["manifest"]).parent
    m = json.loads((release / "MANIFEST.json").read_text())
    check_files(root, m["inputs"])
    check_files(release, m["files"])
    for rel, target in m["attachments"].items():
        if (root / rel).resolve() != (release / target).resolve():
            raise ValueError(f"Attachment mismatch: {rel}")
        if rel != "pose/rows.black_bands.parquet":
            original = pq.read_table((root / rel).with_name("rows.parquet"))
            enriched = pq.read_table(root / rel)
            if not enriched.select(original.column_names).equals(original, check_metadata=False):
                raise ValueError(f"Original rows changed in enriched view: {rel}")
    return {"host": socket.gethostname().split(".")[0], "verified": True,
            "release_id": m["release_id"], "manifest_sha256": sha256(release / "MANIFEST.json"),
            "files_verified": len(m["files"]), "original_inputs_unchanged": len(m["inputs"]),
            "attachments_verified": len(m["attachments"]), "summary": m["summary"]}


README = """# Stage2 black-band annotations

These annotations use the supplied InternalCan/stage2-handover-pose-mimi-sam-v1
shard index: all 133 joints below confidence 0.3, at least 25 consecutive 25 Hz
ticks (1 second). They are inferred from pose confidence, not a new pixel scan.

`conversations.parquet` contains every source conversation, the original raw
intervals, source `shard` and `row_index`, and the new annotations.
`banded_conversations.parquet` is its positive subset.
`banded` reproduces the supplied script (null source lanes count as empty).

Each local store has `rows.black_bands.parquet`, an enriched copy of ALL its
original row columns in exactly the same order. Read it instead of rows.parquet
when you need the new metadata. Existing training loaders are not changed.
`pose/rows.black_bands.parquet` contains the rows selected by POSE_INDEX.json.

- `has_black_band`: true if either lane has a qualifying band overlapping the
  row; false only when both lane annotations are available and empty; null when
  the answer cannot be established. Never coerce null to false when filtering.
- `black_band_status`: band_detected, no_band_detected,
  pose_annotation_unavailable, or source_conversation_missing.
- `black_band_annotation_complete`: both source lane annotations are available.
  A positive row can still be incomplete for its other speaker.
- `speaker_00_black_band_*` / `speaker_01_black_band_*`: availability, flag,
  intervals_seconds, pose_tick_intervals, token_intervals, seconds and n.
- `assistant_has_black_band`, `user_has_black_band` and their interval columns
  follow the local store's explicit speaker mapping.

All new intervals are half-open [start,end), relative to the represented row.
For source conversations, zero is clip_start_sec in the recording. Local store
rows subtract frame_start / 12.5 seconds and clip at n_frames / 12.5. A source
band qualifies BEFORE clipping, so a row can contain less than one second of a
qualifying band at its boundary. Token indices are bounded by row length.
The 25 Hz grid is recovered before floor/ceil conversion to avoid float32
off-by-one errors. Pose ticks are NOT model token indices.

Missing source conversations and null pose annotations remain explicitly
unknown. `unmatched_rows.json` lists every local row absent from the source
index. Xabi V2 uses unrelated recording IDs and is outside this source index.

All original metadata, tokens, manifests, and active training behavior are
preserved. MANIFEST.json records source revision, SHA256 hashes, coverage and
per-store counts. BLACK_BANDS.json in the dataset root locates this release.

Example:

```python
import pyarrow.parquet as pq
rows = pq.read_table('/home/param/datasets/aloe-duplex/s2h-m1-samsep/stores/m1n0/rows.black_bands.parquet')
for row in rows.select(['row_key', 'has_black_band', 'black_band_status',
                        'assistant_black_band_token_intervals',
                        'user_black_band_token_intervals']).to_pylist():
    if row['has_black_band'] is True:
        print(row)
```
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source-root", type=Path, required=True)
    b.add_argument("--dataset-root", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--source-revision", required=True)
    i = sub.add_parser("install")
    i.add_argument("--package", type=Path, required=True)
    i.add_argument("--dataset-root", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--dataset-root", type=Path, required=True)
    a = p.parse_args()
    if a.command == "build":
        result = build(a.source_root, a.dataset_root, a.output, a.source_revision)
        print(json.dumps({"release_id": result["release_id"], "summary": result["summary"]}))
    elif a.command == "install":
        print(json.dumps(install(a.package, a.dataset_root)))
    else:
        print(json.dumps(verify(a.dataset_root)))


if __name__ == "__main__":
    main()
