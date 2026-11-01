#!/usr/bin/env python3
"""Read-only full integrity/structure/finite-value audit of expanded reference bank.

Requires numpy. Example:
 /home/param/vansh-video-models/.venv/bin/python audit_expanded_bank.py
"""
import argparse
import collections
import concurrent.futures
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import struct
import time

import numpy as np

SLOTS = {
    1: "front", 2: "face_image_left", 3: "face_image_right", 4: "mouth_open",
    5: "hands_visible", 6: "hands_interacting", 7: "hair_clothing_01",
    8: "hair_clothing_02", 9: "hair_clothing_03",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def audit_subset(root, name, workers):
    base = root / name
    issues = []
    issue_counts = collections.Counter()

    def issue(kind, detail):
        issue_counts[kind] += 1
        if len(issues) < 200:
            issues.append({"kind": kind, "detail": detail})

    manifest = {}
    for line_no, line in enumerate((base / "SHA256SUMS").read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if not match:
            issue("invalid_manifest_line", line_no)
            continue
        digest, rel = match.groups()
        rel = rel.removeprefix("./")
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            issue("unsafe_manifest_path", rel)
            continue
        if rel in manifest:
            issue("duplicate_manifest_path", rel)
        manifest[rel] = digest

    # Enumerate actual content as well as manifest entries to detect extras.
    actual = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()}
    actual.discard("SHA256SUMS")
    for rel in sorted(actual - manifest.keys()):
        issue("file_not_in_manifest", rel)
    for rel in sorted(manifest.keys() - actual):
        issue("manifest_file_missing", rel)

    groups = collections.defaultdict(list)
    origins = collections.Counter()
    source_types = collections.Counter()
    slot_references = collections.Counter()
    recordings = set()
    sessions = set()
    uuids = set()
    iids = set()
    csv_rows = 0
    previous_key = None
    with (base / "bank.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for row in reader:
            csv_rows += 1
            try:
                uuid = row["uuid"]
                prop = int(row["property"])
                idx = int(row["index"])
                key = (uuid, prop, idx)
                if previous_key is not None and key < previous_key:
                    issue("csv_not_sorted", {"row": csv_rows, "key": key})
                previous_key = key
                if row["slot"] != SLOTS.get(prop):
                    issue("csv_slot_mismatch", {"row": csv_rows, "slot": row["slot"], "property": prop})
                groups[(uuid, prop)].append(idx)
                slot_references[prop] += 1
                uuids.add(uuid)
                recordings.add(row["recording_id"])
                sessions.add(row["session_id"])
                source = row["source"]
                source_type = "gcs_video_provenance" if source.startswith("gs://") else "upstream_relative_png" if source.endswith(".png") else "other"
                source_types[source_type] += 1
                origins[row.get("origin", "not_present")] += 1
                if row["iid"] in iids:
                    issue("duplicate_csv_iid", row["iid"])
                iids.add(row["iid"])
            except Exception as exc:
                issue("csv_invalid_row", {"row": csv_rows, "error": str(exc)})

    for key, indices in groups.items():
        if indices != list(range(len(indices))):
            issue("csv_indices_not_contiguous", {"group": key, "indices": indices})
        if not 1 <= len(indices) <= 20:
            issue("csv_group_count_out_of_range", {"group": key, "n": len(indices)})

    expected_latents = {f"{uuid}/property_{prop}/latents.safetensors" for uuid, prop in groups}
    actual_latents = {rel for rel in actual if rel.endswith(".safetensors")}
    for rel in sorted(expected_latents - actual_latents):
        issue("csv_group_missing_latent", rel)
    for rel in sorted(actual_latents - expected_latents):
        issue("latent_missing_csv_group", rel)

    def check_file(rel):
        errors = []
        size = 0
        n = 0
        nonfinite = 0
        scalar_values = 0
        try:
            data = (base / rel).read_bytes()
            size = len(data)
            digest = hashlib.sha256(data).hexdigest()
            if digest != manifest[rel]:
                errors.append(("sha256_mismatch", {"path": rel, "expected": manifest[rel], "actual": digest}))
            if rel.endswith(".safetensors"):
                if len(data) < 8:
                    raise ValueError("safetensors missing header length")
                header_size = struct.unpack("<Q", data[:8])[0]
                if not 2 <= header_size <= 1024 * 1024 or 8 + header_size > size:
                    raise ValueError(f"invalid header size {header_size}")
                header = json.loads(data[8:8 + header_size])
                if set(header) != {"__metadata__", "video"}:
                    errors.append(("unexpected_tensor_keys", {"path": rel, "keys": list(header)}))
                tensor = header["video"]
                shape = tensor["shape"]
                if len(shape) != 4 or shape[1:] != [48, 30, 16] or not 1 <= shape[0] <= 20:
                    errors.append(("invalid_video_shape", {"path": rel, "shape": shape}))
                n = shape[0]
                if tensor["dtype"] != "BF16":
                    errors.append(("invalid_video_dtype", {"path": rel, "dtype": tensor["dtype"]}))
                parts = Path(rel).parts
                if len(parts) != 3 or parts[2] != "latents.safetensors" or not re.fullmatch(r"property_[1-9]", parts[1]):
                    raise ValueError("unexpected latent path")
                uuid, prop = parts[0], int(parts[1][9:])
                expected_meta = {"uuid": uuid, "property": str(prop), "slot": SLOTS[prop], "n": str(n)}
                for k, v in expected_meta.items():
                    if header.get("__metadata__", {}).get(k) != v:
                        errors.append(("metadata_mismatch", {"path": rel, "key": k, "expected": v, "actual": header.get("__metadata__", {}).get(k)}))
                if len(groups.get((uuid, prop), [])) != n:
                    errors.append(("csv_latent_count_mismatch", {"path": rel, "latent_n": n, "csv_n": len(groups.get((uuid, prop), []))}))
                expected_bytes = int(np.prod(shape)) * 2
                if tensor["data_offsets"] != [0, expected_bytes] or size != 8 + header_size + expected_bytes:
                    errors.append(("tensor_payload_size_mismatch", {"path": rel, "offsets": tensor["data_offsets"], "file_size": size, "expected_size": 8 + header_size + expected_bytes}))
                else:
                    values = np.frombuffer(data, dtype="<u2", offset=8 + header_size)
                    scalar_values = values.size
                    # BF16 shares float32's 8-bit exponent: all 1s means NaN/Inf.
                    nonfinite = int(np.count_nonzero((values & 0x7F80) == 0x7F80))
                    if nonfinite:
                        errors.append(("nonfinite_tensor_values", {"path": rel, "count": nonfinite}))
        except Exception as exc:
            errors.append(("file_check_exception", {"path": rel, "error": f"{type(exc).__name__}: {exc}"}))
        return size, n, nonfinite, scalar_values, errors

    print(f"{name}: CSV rows={csv_rows:,}; checking {len(manifest):,} manifest files with {workers} threads", flush=True)
    total_bytes = 0
    latent_images = 0
    nonfinite_values = 0
    scalar_values = 0
    n_histogram = collections.Counter()
    completed = 0
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for size, n, nonfinite, nvalues, errors in executor.map(check_file, sorted(manifest)):
            total_bytes += size
            if n:
                n_histogram[n] += 1
                latent_images += n
            nonfinite_values += nonfinite
            scalar_values += nvalues
            for kind, detail in errors:
                issue(kind, detail)
            completed += 1
            if completed % 1000 == 0:
                print(f"{name}: checked {completed:,}/{len(manifest):,}; errors={sum(issue_counts.values())}; elapsed={time.monotonic()-started:.1f}s", flush=True)

    per_person = collections.Counter(uuid for uuid, prop in groups)
    slot_people = collections.Counter(prop for uuid, prop in groups)
    return {
        "subset": name,
        "manifest_sha256": sha256(base / "SHA256SUMS"),
        "csv_sha256": sha256(base / "bank.csv"),
        "csv_columns": columns,
        "manifest_files": len(manifest),
        "actual_files_excluding_manifest": len(actual),
        "checked_file_bytes": total_bytes,
        "latent_files": len(actual_latents),
        "latent_images": latent_images,
        "csv_rows": csv_rows,
        "people": len(uuids),
        "recordings": len(recordings),
        "sessions": len(sessions),
        "origins": dict(origins),
        "source_types": dict(source_types),
        "slot_distribution": {str(prop): {"slot": SLOTS[prop], "people_with_slot": slot_people[prop], "references": slot_references[prop], "people_missing_slot": len(uuids)-slot_people[prop]} for prop in SLOTS},
        "slots_per_person_histogram": dict(sorted(collections.Counter(per_person.values()).items())),
        "images_per_present_slot_histogram": dict(sorted(n_histogram.items())),
        "finite_bf16_scalars_checked": scalar_values,
        "nonfinite_bf16_scalars": nonfinite_values,
        "issue_count": sum(issue_counts.values()),
        "issue_counts": dict(issue_counts),
        "first_200_issues": issues,
        "payload_audit_seconds": round(time.monotonic()-started, 3),
        "passed": not issue_counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/paramfs/param/outputs/reference-compression/data/reference-bank-256x480-muon-encoded"))
    parser.add_argument("--local-copy", type=Path, default=Path("/home/param/reference-compression-data/reference-bank-256x480-muon-encoded"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("expanded-bank-audit.json"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = time.monotonic()
    report = {"root": str(args.root), "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "workers": args.workers, "subsets": {}}
    report["recipe_sha256"] = sha256(args.root / "RECIPE.md")
    report["sinks_sha256"] = sha256(args.root / "sinks.md")
    report["local_copy_comparison"] = {}
    for rel in ["RECIPE.md", "sinks.md", "bank/SHA256SUMS", "bank/bank.csv", "beatles-eval/SHA256SUMS", "beatles-eval/bank.csv"]:
        remote_hash = sha256(args.root / rel)
        local_path = args.local_copy / rel
        local_hash = sha256(local_path) if local_path.is_file() else None
        report["local_copy_comparison"][rel] = {"paramfs_sha256": remote_hash, "local_sha256": local_hash, "equal": remote_hash == local_hash}
    for subset in ["bank", "beatles-eval"]:
        report["subsets"][subset] = audit_subset(args.root, subset, args.workers)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["completed_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["elapsed_seconds"] = round(time.monotonic()-started, 3)
    report["passed"] = all(s["passed"] for s in report["subsets"].values())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "passed": report["passed"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
