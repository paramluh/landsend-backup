#!/usr/bin/env python3
"""Build split datasets with Mimi + pose + question + answer.

Pipeline per split:
1) Filter speech-token parquet rows by split clip_ids.
2) Run pose-LLM-trainer/build_chunk_token_dataset.py to attach pose tokens.
3) Add question/answer text columns:
   - question: from tiktok_questions clip_id mapping
   - answer:   chunk transcript from word_texts
4) Collate into interleaved input_ids while preserving question/answer metadata.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
OUT_ROOT = Path(__file__).resolve().parent / "collated_with_mimi_questions_answers"
POSE_LLM_ROOT = Path("/home/james/pose-LLM-trainer")
BUILD_CHUNK_SCRIPT = POSE_LLM_ROOT / "build_chunk_token_dataset.py"
DEFAULT_COLLATE_SCRIPT = POSE_LLM_ROOT / "collate.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("with_objects", "without_objects", "both"),
        default="both",
    )
    parser.add_argument(
        "--resources-dir",
        default=str(RESOURCES_DIR),
        help=f"Resources dir from prepare_with_without_resources.py (default: {RESOURCES_DIR}).",
    )
    parser.add_argument(
        "--speech-input-glob",
        default="/mnt/som-fs/pose_cond/audio_tokenized/tiktok-pipeline-merged/*.parquet",
        help="Input speech-token parquet glob (mimi + word timestamps).",
    )
    parser.add_argument(
        "--questions-parquet",
        default="/mnt/som-fs/pose_cond/tiktok_questions/tiktok_clip_questions.parquet",
        help="Parquet with clip_id->question mapping.",
    )
    parser.add_argument(
        "--questions-source",
        choices=("auto", "resource_tsv", "parquet"),
        default="auto",
        help=(
            "Question mapping source. "
            "'resource_tsv' reads <split>_question_answer.tsv from resources, "
            "'parquet' reads --questions-parquet, 'auto' prefers resource TSV then parquet."
        ),
    )
    parser.add_argument(
        "--pose-dir",
        default="/mnt/som-fs/pose_cond/pose_cond_npy",
        help="Local pose npy directory used by build_chunk_token_dataset.py.",
    )
    parser.add_argument(
        "--dims-csv",
        default="/mnt/som-fs/pose_cond/dims_upperbody_ideal_fresh.csv",
        help="dims csv passed to build_chunk_token_dataset.py.",
    )
    parser.add_argument(
        "--pose-tokenizer",
        default="xabirizar9/pose-tokenizer-2x-12cb-lossshift1",
        help="Pose tokenizer id/path passed to build_chunk_token_dataset.py.",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=3,
        help="Lookahead forwarded to collation.",
    )
    parser.add_argument(
        "--num-pose-codebooks",
        type=int,
        default=12,
        help="Number of pose codebooks to collate.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Process count for datasets map in collation.",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=8192,
        help="Batch size for parquet streaming filters/enrichment.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUT_ROOT),
        help=f"Output root (default: {OUT_ROOT}).",
    )
    parser.add_argument(
        "--build-extra-args",
        default="",
        help=(
            "Extra args appended to build_chunk_token_dataset.py. "
            "Example: \"--workers-per-gpu 2 --num-gpus 8\""
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional debug cap for collation output rows.",
    )
    parser.add_argument(
        "--collate-script",
        default=str(DEFAULT_COLLATE_SCRIPT),
        help=(
            "Path to collation script to import. "
            "Use collate.py for 12cb/1024 pose codebooks or collation_32.py for 32cb/2048."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild outputs even if step outputs already exist.",
    )
    return parser.parse_args()


def _split_names(split: str) -> list[str]:
    if split == "both":
        return ["with_objects", "without_objects"]
    return [split]


def _clip_ids_path(resources_dir: Path, split_name: str) -> Path:
    if split_name == "with_objects":
        return resources_dir / "with_objects_clip_ids.txt"
    return resources_dir / "without_objects_clip_ids.txt"


def _load_clip_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing clip id file: {path}")
    with path.open(encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _filter_speech_parquets(
    speech_glob: str,
    clip_ids: set[str],
    output_dir: Path,
    read_batch_size: int,
) -> tuple[int, int]:
    input_files = sorted(glob.glob(speech_glob))
    if not input_files:
        raise FileNotFoundError(f"No speech parquet matched {speech_glob!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    shard_idx = 0
    rows_in = 0
    rows_out = 0

    def flush_rows(rows: list[dict[str, Any]]) -> None:
        nonlocal writer, shard_idx, rows_out
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if writer is None:
            out_path = output_dir / f"part-{shard_idx:05d}.parquet"
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd", compression_level=3)
        writer.write_table(table)
        rows_out += table.num_rows

    buffer: list[dict[str, Any]] = []
    for path in input_files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=read_batch_size):
            rows = batch.to_pylist()
            rows_in += len(rows)
            for row in rows:
                if row.get("clip_id") in clip_ids:
                    buffer.append(row)
            if len(buffer) >= read_batch_size:
                flush_rows(buffer)
                buffer.clear()

    if buffer:
        flush_rows(buffer)
        buffer.clear()
    if writer is not None:
        writer.close()
        writer = None
        shard_idx += 1

    if rows_out == 0:
        raise RuntimeError("Filtered speech rows are empty; check clip ids vs speech input.")
    return rows_in, rows_out


def _run_build_chunk_dataset(
    *,
    filtered_speech_glob: str,
    output_dir: Path,
    pose_dir: str,
    dims_csv: str,
    pose_tokenizer: str,
    extra_args: str,
) -> None:
    cmd = [
        sys.executable,
        str(BUILD_CHUNK_SCRIPT),
        "--input",
        filtered_speech_glob,
        "--output-dir",
        str(output_dir),
        "--pose-dir",
        pose_dir,
        "--dims-csv",
        dims_csv,
        "--pose-tokenizer",
        pose_tokenizer,
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _build_question_map(
    *,
    split_name: str,
    resources_dir: Path,
    source_mode: str,
    questions_parquet: Path,
    needed_clip_ids: set[str],
) -> dict[str, tuple[str, bool]]:
    tsv_name = f"{split_name}_question_answer.tsv"
    tsv_path = resources_dir / tsv_name

    def from_tsv(path: Path) -> dict[str, tuple[str, bool]]:
        found: dict[str, tuple[str, bool]] = {}
        if not path.is_file():
            return found
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cid = str(row.get("clip_id") or "")
                if cid in needed_clip_ids and cid and cid not in found:
                    matched = str(row.get("matched") or "").strip().lower() in {
                        "true",
                        "1",
                        "t",
                        "yes",
                    }
                    found[cid] = (str(row.get("question") or ""), matched)
        return found

    def from_parquet(path: Path) -> dict[str, tuple[str, bool]]:
        if not path.is_file():
            raise FileNotFoundError(f"Questions parquet not found: {path}")
        found: dict[str, tuple[str, bool]] = {}
        parquet = pq.ParquetFile(path)
        cols = parquet.schema_arrow.names
        read_cols = [c for c in ["clip_id", "question", "matched"] if c in cols]
        if "clip_id" not in read_cols or "question" not in read_cols:
            raise ValueError(f"{path} missing clip_id/question columns")
        for batch in parquet.iter_batches(batch_size=100_000, columns=read_cols):
            data = batch.to_pydict()
            clip_ids = data.get("clip_id", [])
            questions = data.get("question", [])
            matcheds = data.get("matched", [False] * len(clip_ids))
            for clip_id, question, matched in zip(clip_ids, questions, matcheds):
                cid = str(clip_id)
                if cid in needed_clip_ids and cid not in found:
                    found[cid] = (str(question or ""), bool(matched))
        return found

    if source_mode == "resource_tsv":
        qmap = from_tsv(tsv_path)
        if not qmap:
            raise FileNotFoundError(f"Question TSV missing or empty: {tsv_path}")
        return qmap
    if source_mode == "parquet":
        return from_parquet(questions_parquet)

    # auto: prefer split-specific resource TSV, fallback to parquet
    qmap = from_tsv(tsv_path)
    if qmap:
        return qmap
    return from_parquet(questions_parquet)


def _answer_from_word_texts(row: dict[str, Any]) -> str:
    words = row.get("word_texts") or []
    if not words:
        return ""
    return " ".join(str(w) for w in words).strip()


def _enrich_chunk_rows_with_qa(
    chunk_glob: str,
    question_map: dict[str, tuple[str, bool]],
    output_dir: Path,
    read_batch_size: int,
) -> int:
    input_files = sorted(glob.glob(chunk_glob))
    if not input_files:
        raise FileNotFoundError(f"No chunk token parquet matched {chunk_glob!r}")
    output_dir.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    shard_idx = 0
    rows_written = 0

    def write_rows(rows: list[dict[str, Any]]) -> None:
        nonlocal writer, rows_written, shard_idx
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if writer is None:
            out_path = output_dir / f"part-{shard_idx:05d}.parquet"
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd", compression_level=3)
        writer.write_table(table)
        rows_written += table.num_rows

    buffer: list[dict[str, Any]] = []
    for path in input_files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=read_batch_size):
            for row in batch.to_pylist():
                clip_id = str(row.get("clip_id") or "")
                question, matched = question_map.get(clip_id, ("", False))
                row["question"] = question
                row["answer"] = _answer_from_word_texts(row)
                row["question_matched"] = matched
                buffer.append(row)
            if len(buffer) >= read_batch_size:
                write_rows(buffer)
                buffer.clear()

    if buffer:
        write_rows(buffer)
        buffer.clear()
    if writer is not None:
        writer.close()
        writer = None
    return rows_written


def _load_collate_module(collate_script: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pose_llm_collate", collate_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {collate_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collate_enriched_rows(
    *,
    enriched_glob: str,
    output_dir: Path,
    collate_script: Path,
    lookahead: int,
    num_pose_codebooks: int,
    num_proc: int,
    max_rows: int,
) -> Path:
    data_files = sorted(glob.glob(enriched_glob))
    if not data_files:
        raise FileNotFoundError(f"No enriched parquet matched {enriched_glob!r}")

    collate_mod = _load_collate_module(collate_script)
    for col in ["question", "answer", "question_matched"]:
        if col not in collate_mod.METADATA_COLUMNS:
            collate_mod.METADATA_COLUMNS.append(col)

    ds = load_dataset("parquet", data_files=data_files, split="train")
    if max_rows > 0:
        ds = ds.select(range(min(max_rows, ds.num_rows)))

    ds = ds.map(
        collate_mod.build_example,
        fn_kwargs={"lookahead": lookahead, "num_pose_codebooks": num_pose_codebooks},
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="collate text+speech+pose (+qa metadata)",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "part-00000.parquet"
    ds.to_parquet(str(out_path))
    return out_path


def _run_split(args: argparse.Namespace, split_name: str) -> None:
    resources_dir = Path(args.resources_dir).resolve()
    out_root = Path(args.output_root).resolve()
    split_root = out_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    clip_ids_file = _clip_ids_path(resources_dir, split_name)
    clip_ids = _load_clip_ids(clip_ids_file)
    print(f"[{split_name}] clip_ids={len(clip_ids):,} from {clip_ids_file}")

    speech_filtered_dir = split_root / "speech_filtered"
    chunk_tokens_dir = split_root / "chunk_audio_pose_tokens"
    chunk_enriched_dir = split_root / "chunk_audio_pose_tokens_with_qa"
    collated_dir = split_root / "collated_input_ids_with_qa"

    if args.overwrite and split_root.exists():
        # keep parent directory but remove step outputs to avoid stale mix
        for step_dir in [speech_filtered_dir, chunk_tokens_dir, chunk_enriched_dir, collated_dir]:
            if step_dir.exists():
                for p in step_dir.glob("**/*"):
                    if p.is_file():
                        p.unlink()

    speech_glob = str(speech_filtered_dir / "*.parquet")
    if not sorted(glob.glob(speech_glob)):
        rows_in, rows_out = _filter_speech_parquets(
            args.speech_input_glob,
            clip_ids,
            speech_filtered_dir,
            args.read_batch_size,
        )
        print(f"[{split_name}] speech filter: in={rows_in:,}, out={rows_out:,} -> {speech_filtered_dir}")
    else:
        print(f"[{split_name}] speech filter: using existing {speech_filtered_dir}")

    chunk_glob = str(chunk_tokens_dir / "*.parquet")
    if not sorted(glob.glob(chunk_glob)):
        _run_build_chunk_dataset(
            filtered_speech_glob=speech_glob,
            output_dir=chunk_tokens_dir,
            pose_dir=args.pose_dir,
            dims_csv=args.dims_csv,
            pose_tokenizer=args.pose_tokenizer,
            extra_args=args.build_extra_args,
        )
        print(f"[{split_name}] built chunk tokens -> {chunk_tokens_dir}")
    else:
        print(f"[{split_name}] chunk tokens: using existing {chunk_tokens_dir}")

    enriched_glob = str(chunk_enriched_dir / "*.parquet")
    if not sorted(glob.glob(enriched_glob)):
        question_map = _build_question_map(
            split_name=split_name,
            resources_dir=resources_dir,
            source_mode=args.questions_source,
            questions_parquet=Path(args.questions_parquet).resolve(),
            needed_clip_ids=clip_ids,
        )
        written = _enrich_chunk_rows_with_qa(
            chunk_glob=str(chunk_tokens_dir / "*.parquet"),
            question_map=question_map,
            output_dir=chunk_enriched_dir,
            read_batch_size=args.read_batch_size,
        )
        print(f"[{split_name}] enriched rows={written:,} -> {chunk_enriched_dir}")
    else:
        print(f"[{split_name}] enriched chunk rows: using existing {chunk_enriched_dir}")

    out_path = _collate_enriched_rows(
        enriched_glob=str(chunk_enriched_dir / "*.parquet"),
        output_dir=collated_dir,
        collate_script=Path(args.collate_script).resolve(),
        lookahead=args.lookahead,
        num_pose_codebooks=args.num_pose_codebooks,
        num_proc=args.num_proc,
        max_rows=args.max_rows,
    )
    print(f"[{split_name}] collated output -> {out_path}")


def main() -> int:
    args = parse_args()
    for split_name in _split_names(args.split):
        _run_split(args, split_name)
    print("\nDone.")
    print(f"output_root={Path(args.output_root).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
