#!/usr/bin/env python3
"""Build and upload split QA+Mimi-token datasets to Hugging Face.

For each split (with_objects / without_objects), this script:
1) Loads split clip IDs from resources.
2) Filters local speech-token parquet rows by clip_id.
3) Attaches question metadata from resources/<split>_question_answer.tsv.
4) Writes split parquet shards locally.
5) Pushes the split dataset to a target HF repo.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESOURCES = Path(__file__).resolve().parent / "resources"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "hf_upload_ready_qa_mimi_tokens"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("with_objects", "without_objects", "both"),
        default="both",
    )
    parser.add_argument(
        "--resources-dir",
        default=str(DEFAULT_RESOURCES),
        help=f"Split resources directory (default: {DEFAULT_RESOURCES}).",
    )
    parser.add_argument(
        "--speech-input-glob",
        default="/mnt/som-fs/pose_cond/audio_tokenized/tiktok-pipeline-merged/*.parquet",
        help="Input speech-token parquet glob.",
    )
    parser.add_argument(
        "--repo-with",
        default="InternalCan/qa_mimi_tokens_object",
        help="HF repo for with_objects split.",
    )
    parser.add_argument(
        "--repo-without",
        default="InternalCan/qa_mimi_tokens_no_object",
        help="HF repo for without_objects split.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT),
        help=f"Local output root before upload (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument("--read-batch-size", type=int, default=100_000)
    parser.add_argument("--shard-rows", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(private=True)
    parser.add_argument("--private", dest="private", action="store_true", help="Push private repos (default).")
    parser.add_argument("--public", dest="private", action="store_false", help="Push as public.")
    return parser.parse_args()


def _split_names(split: str) -> list[str]:
    if split == "both":
        return ["with_objects", "without_objects"]
    return [split]


def _clip_ids_path(resources_dir: Path, split_name: str) -> Path:
    return resources_dir / ("with_objects_clip_ids.txt" if split_name == "with_objects" else "without_objects_clip_ids.txt")


def _qa_tsv_path(resources_dir: Path, split_name: str) -> Path:
    return resources_dir / (
        "with_objects_question_answer.tsv"
        if split_name == "with_objects"
        else "without_objects_question_answer.tsv"
    )


def _load_clip_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing clip id file: {path}")
    with path.open(encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _load_qa_map(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cid = str(row.get("clip_id") or "")
            if not cid:
                continue
            out[cid] = {
                "question": str(row.get("question") or ""),
                "clip_transcript": str(row.get("answer") or ""),
                "question_matched": str(row.get("matched") or "").strip().lower()
                in {"true", "1", "t", "yes"},
            }
    return out


def _answer_from_words(row: dict[str, Any]) -> str:
    words = row.get("word_texts") or []
    if not words:
        return ""
    return " ".join(str(w) for w in words).strip()


def _write_rows_sharded(
    rows_iter: Any,
    output_dir: Path,
    shard_rows: int,
) -> tuple[int, int]:
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    shards_written = 0
    buf: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal writer, rows_written, shards_written, buf
        if not buf:
            return
        table = pa.Table.from_pylist(buf)
        if writer is None:
            out_path = output_dir / f"part-{shards_written:05d}.parquet"
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd", compression_level=3)
            shards_written += 1
        writer.write_table(table)
        rows_written += table.num_rows
        buf = []

    for row in rows_iter:
        buf.append(row)
        if len(buf) >= shard_rows:
            flush()
            if writer is not None:
                writer.close()
                writer = None
    flush()
    if writer is not None:
        writer.close()
    return rows_written, shards_written


def _build_split_parquet(
    *,
    split_name: str,
    resources_dir: Path,
    speech_input_glob: str,
    output_dir: Path,
    read_batch_size: int,
    shard_rows: int,
) -> tuple[int, int, int]:
    clip_ids = _load_clip_ids(_clip_ids_path(resources_dir, split_name))
    qa_map = _load_qa_map(_qa_tsv_path(resources_dir, split_name))
    input_files = sorted(glob.glob(speech_input_glob))
    if not input_files:
        raise FileNotFoundError(f"No speech parquet matched {speech_input_glob!r}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_scanned = 0
    has_objects = split_name == "with_objects"

    def row_iter():
        nonlocal rows_scanned
        for path in input_files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=read_batch_size):
                clip_col = batch.column("clip_id").to_pylist()
                rows_scanned += len(clip_col)
                keep_idx = [i for i, cid in enumerate(clip_col) if cid in clip_ids]
                if not keep_idx:
                    continue
                subset = batch.take(pa.array(keep_idx, type=pa.int32()))
                for row in subset.to_pylist():
                    clip_id = row.get("clip_id")
                    qa = qa_map.get(clip_id, {})
                    yield {
                        "row_id": row.get("row_id"),
                        "video_id": row.get("video_id"),
                        "clip_id": clip_id,
                        "clip_index": row.get("clip_index"),
                        "chunk_index": row.get("chunk_index"),
                        "chunk_id": row.get("chunk_id"),
                        "chunk_start_seconds": row.get("chunk_start_seconds"),
                        "chunk_end_seconds": row.get("chunk_end_seconds"),
                        "chunk_duration_seconds": row.get("chunk_duration_seconds"),
                        "num_frames": row.get("num_frames"),
                        "mimi_codes": row.get("mimi_codes"),
                        "speech_tokens": row.get("mimi_codes"),
                        "question": qa.get("question", ""),
                        "answer": _answer_from_words(row),
                        "clip_transcript": qa.get("clip_transcript", ""),
                        "question_matched": bool(qa.get("question_matched", False)),
                        "has_objects": has_objects,
                    }

    rows_written, shards_written = _write_rows_sharded(row_iter(), output_dir, shard_rows)
    return rows_scanned, rows_written, shards_written


def _push_split_to_hub(split_dir: Path, repo: str, private: bool) -> int:
    files = sorted(glob.glob(str(split_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No split parquet files found in {split_dir}")
    ds = load_dataset("parquet", data_files=files, split="train")
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("hf_token")
        or os.environ.get("xabi_token")
        or os.environ.get("canopy_token")
    )
    ds.push_to_hub(repo, private=private, token=token)
    return ds.num_rows


def main() -> int:
    args = parse_args()
    resources_dir = Path(args.resources_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    repo_by_split = {
        "with_objects": args.repo_with,
        "without_objects": args.repo_without,
    }

    for split_name in _split_names(args.split):
        split_dir = output_root / split_name
        if split_dir.exists() and args.overwrite:
            for p in split_dir.glob("*.parquet"):
                p.unlink()
        scanned, written, shards = _build_split_parquet(
            split_name=split_name,
            resources_dir=resources_dir,
            speech_input_glob=args.speech_input_glob,
            output_dir=split_dir,
            read_batch_size=args.read_batch_size,
            shard_rows=args.shard_rows,
        )
        print(
            f"[{split_name}] scanned={scanned:,}, written={written:,}, shards={shards} -> {split_dir}",
            flush=True,
        )
        repo = repo_by_split[split_name]
        pushed_rows = _push_split_to_hub(split_dir, repo, private=args.private)
        print(
            f"[{split_name}] pushed rows={pushed_rows:,} -> {repo} (private={args.private})",
            flush=True,
        )

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
