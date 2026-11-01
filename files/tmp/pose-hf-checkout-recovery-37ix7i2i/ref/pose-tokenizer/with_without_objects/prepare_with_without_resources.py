#!/usr/bin/env python3
"""Prepare with/without-object split resources for pose tokenization.

This script writes split CSVs and manifest files under:
  with_without_objects/resources/

Outputs:
  - <bucket>_all.csv
  - <bucket>_with_objects.csv
  - <bucket>_without_objects.csv
  - with_objects_clip_ids.txt
  - without_objects_clip_ids.txt
  - with_objects_pose_manifest.txt
  - without_objects_pose_manifest.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_poses import CSV_HEADER, export_paths, read_rows_csv, write_rows_csv  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket",
        default="hands",
        help="DB bucket to export when --source-csv is not provided (default: hands).",
    )
    parser.add_argument(
        "--source-csv",
        default="",
        help=(
            "Optional existing all-rows CSV with columns "
            "clip_id,gcs_clip_path,gcs_pose_path,has_objects. "
            "If set, DB is not queried."
        ),
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Minimum duration filter when querying DB (default: 3.0).",
    )
    parser.add_argument(
        "--resources-dir",
        default=str(RESOURCES_DIR),
        help=f"Where to write outputs (default: {RESOURCES_DIR}).",
    )
    parser.add_argument(
        "--questions-parquet",
        default="/mnt/som-fs/pose_cond/tiktok_questions/tiktok_clip_questions.parquet",
        help="Parquet containing clip_id->question mapping.",
    )
    return parser.parse_args()


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "t", "yes"}


def _write_clip_id_and_manifest(
    rows: list[tuple[str, str, str, bool]],
    clip_ids_path: Path,
    manifest_path: Path,
) -> None:
    with clip_ids_path.open("w", encoding="utf-8") as f_clip, manifest_path.open(
        "w", encoding="utf-8"
    ) as f_manifest:
        for clip_id, _, _, _ in rows:
            f_clip.write(f"{clip_id}\n")
            f_manifest.write(f"{clip_id}_poses.npy\n")


def _build_question_map(
    questions_parquet: Path,
    needed_clip_ids: set[str],
) -> dict[str, tuple[str, str, bool]]:
    """Return clip_id -> (question, transcript, matched)."""
    if not questions_parquet.is_file():
        return {}
    found: dict[str, tuple[str, str, bool]] = {}
    parquet = pq.ParquetFile(questions_parquet)
    cols = parquet.schema_arrow.names
    read_cols = [c for c in ["clip_id", "question", "transcript", "matched"] if c in cols]
    if "clip_id" not in read_cols or "question" not in read_cols:
        return {}
    for batch in parquet.iter_batches(batch_size=100_000, columns=read_cols):
        data = batch.to_pydict()
        clip_ids = data.get("clip_id", [])
        questions = data.get("question", [])
        transcripts = data.get("transcript", [""] * len(clip_ids))
        matcheds = data.get("matched", [False] * len(clip_ids))
        for clip_id, question, transcript, matched in zip(
            clip_ids, questions, transcripts, matcheds
        ):
            cid = str(clip_id)
            if cid in needed_clip_ids and cid not in found:
                found[cid] = (
                    str(question or ""),
                    str(transcript or ""),
                    bool(matched),
                )
    return found


def _write_questions_txt(
    rows: list[tuple[str, str, str, bool]],
    question_map: dict[str, tuple[str, str, bool]],
    questions_txt_path: Path,
    qa_tsv_path: Path,
) -> tuple[int, int]:
    """Write split question files.

    - questions_txt_path: one question per line (clip order).
    - qa_tsv_path: clip_id, question, answer(transcript), matched.
    """
    q_written = 0
    qa_written = 0
    with questions_txt_path.open("w", encoding="utf-8") as fq, qa_tsv_path.open(
        "w", encoding="utf-8", newline=""
    ) as fqa:
        writer = csv.writer(fqa, delimiter="\t")
        writer.writerow(["clip_id", "question", "answer", "matched"])
        for clip_id, _, _, _ in rows:
            entry = question_map.get(clip_id)
            if entry is None:
                continue
            question, transcript, matched = entry
            fq.write(f"{question}\n")
            writer.writerow([clip_id, question, transcript, "true" if matched else "false"])
            q_written += 1
            qa_written += 1
    return q_written, qa_written


def _read_source_rows(path: Path) -> list[tuple[str, str, str, bool]]:
    rows = read_rows_csv(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _write_split_csv(path: Path, rows: list[tuple[str, str, str, bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for clip_id, gcs_clip_path, gcs_pose_path, has_objects in rows:
            writer.writerow(
                [
                    clip_id,
                    gcs_clip_path,
                    gcs_pose_path,
                    "true" if has_objects else "false",
                ]
            )


def main() -> int:
    args = parse_args()
    resources_dir = Path(args.resources_dir).resolve()
    resources_dir.mkdir(parents=True, exist_ok=True)

    if args.source_csv:
        src = Path(args.source_csv).resolve()
        print(f"Loading rows from {src}")
        rows = _read_source_rows(src)
        bucket_label = args.bucket or src.stem
    else:
        print(f"Querying DB bucket={args.bucket!r}, min_duration={args.min_duration}")
        rows = export_paths(args.bucket, min_duration=args.min_duration)
        bucket_label = args.bucket
        if not rows:
            print("No rows returned from DB query.")
            return 1

    all_csv = resources_dir / f"{bucket_label}_all.csv"
    with_csv = resources_dir / f"{bucket_label}_with_objects.csv"
    without_csv = resources_dir / f"{bucket_label}_without_objects.csv"

    with_rows = [row for row in rows if row[3]]
    without_rows = [row for row in rows if not row[3]]

    write_rows_csv(all_csv, rows)
    _write_split_csv(with_csv, with_rows)
    _write_split_csv(without_csv, without_rows)

    with_clip_ids = resources_dir / "with_objects_clip_ids.txt"
    without_clip_ids = resources_dir / "without_objects_clip_ids.txt"
    with_manifest = resources_dir / "with_objects_pose_manifest.txt"
    without_manifest = resources_dir / "without_objects_pose_manifest.txt"

    _write_clip_id_and_manifest(with_rows, with_clip_ids, with_manifest)
    _write_clip_id_and_manifest(without_rows, without_clip_ids, without_manifest)

    all_clip_ids = {clip_id for clip_id, _, _, _ in rows}
    question_map = _build_question_map(Path(args.questions_parquet).resolve(), all_clip_ids)
    with_q_txt = resources_dir / "with_objects_questions.txt"
    without_q_txt = resources_dir / "without_objects_questions.txt"
    with_qa_tsv = resources_dir / "with_objects_question_answer.tsv"
    without_qa_tsv = resources_dir / "without_objects_question_answer.tsv"
    with_q_count, _ = _write_questions_txt(with_rows, question_map, with_q_txt, with_qa_tsv)
    without_q_count, _ = _write_questions_txt(
        without_rows, question_map, without_q_txt, without_qa_tsv
    )

    print(f"all_rows={len(rows):,} -> {all_csv}")
    print(f"with_objects_rows={len(with_rows):,} -> {with_csv}")
    print(f"without_objects_rows={len(without_rows):,} -> {without_csv}")
    print(f"with_manifest={with_manifest}")
    print(f"without_manifest={without_manifest}")
    if question_map:
        print(f"with_questions_rows={with_q_count:,} -> {with_q_txt}")
        print(f"without_questions_rows={without_q_count:,} -> {without_q_txt}")
        print(f"with_qa_tsv={with_qa_tsv}")
        print(f"without_qa_tsv={without_qa_tsv}")
    else:
        print("question mapping unavailable; skipped question txt/tsv export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
