"""Tokenize full pose clips and save decoded reconstruction chunks.

This sibling of ``tokenize_chunks.py`` keeps the token export contract intact
while also writing the corresponding normalized reconstructed pose chunks.

Per segment we:

1. Read one or more manifest files with entries such as
   ``v2/<clip_id>/<segment>``.
2. Load/preprocess/resample poses exactly like ``tokenize_chunks.py``.
3. Drop frame 0, matching the Wan convention where each 77-frame chunk has one
   shared conditioning frame followed by 76 generated frames.
4. Tokenize the full post-skip clip in one pass.
5. Decode those tokens back to normalized pose features in one pass.
6. Split tokens into parquet rows and reconstructed poses into one
   ``.safetensors`` file per chunk.

Token parquet rows match ``tokenize_chunks.py``:

    clip_id     string
    chunk_path  string
    tokens      fixed_size_list<int16>[n_codebooks * n_token_frames]

Reconstruction files are saved beside the existing Wan chunks by default,
under ``<wan-root>/<clip_path>/<segment>_poses_chunkXXXX.safetensors``. Each
file contains:

    reconstruction  float32 tensor shaped (chunk_frames, feature_dim)

The reconstruction is the model's normalized output feature space; this script
does not convert back to shoulder-width or image-resolution coordinates.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from pose_tokenizer import PoseTokenizerModel
from pose_tokenizer.config import TrainConfig
from pose_tokenizer.data.kinematic import NUM_JOINTS

try:
    from scripts.tokenize_chunks import (
        ChunkedPoseDataset,
        _scan_pose_dir,
        collate_passthrough,
        is_hf_repo_id,
        resolve_model_and_config as _resolve_model_and_config,
    )
except ModuleNotFoundError:
    from tokenize_chunks import (
        ChunkedPoseDataset,
        _scan_pose_dir,
        collate_passthrough,
        is_hf_repo_id,
        resolve_model_and_config as _resolve_model_and_config,
    )


@dataclass(frozen=True)
class ManifestEntry:
    version: str | None
    clip_id: str
    segment: str
    clip_path: str


def _entry_from_path(raw_path: str) -> ManifestEntry | None:
    line = raw_path.strip().strip("\"'").rstrip("/")
    if not line or line in {"paths", "path", "clip_path"}:
        return None

    name = Path(line).name
    if name.endswith((".npy", ".npz")):
        segment = re.sub(r"(_poses)?\.(npy|npz)$", "", name)
        clip_id = segment.split("_", 1)[0]
        return ManifestEntry(None, clip_id, segment, segment)

    parts = line.split("/")
    if len(parts) >= 3 and re.fullmatch(r"v\d+", parts[-3]):
        version = parts[-3]
        clip_id = parts[-2]
        segment = parts[-1]
        return ManifestEntry(version, clip_id, segment, f"{version}/{clip_id}/{segment}")
    if len(parts) >= 2:
        clip_id = parts[-2]
        segment = parts[-1]
        return ManifestEntry(None, clip_id, segment, f"{clip_id}/{segment}")
    return None


def _parse_clip_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """Read CSV/TXT manifests into entries that preserve v2/v3/v4 paths."""
    entries: list[ManifestEntry] = []
    if manifest_path.suffix.lower() == ".csv":
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                preferred = (
                    "clip_path"
                    if "clip_path" in reader.fieldnames
                    else "path"
                    if "path" in reader.fieldnames
                    else "paths"
                    if "paths" in reader.fieldnames
                    else reader.fieldnames[0]
                )
                for row in reader:
                    entry = _entry_from_path(row.get(preferred, ""))
                    if entry is not None:
                        entries.append(entry)
            else:
                f.seek(0)
                for raw in f:
                    entry = _entry_from_path(raw)
                    if entry is not None:
                        entries.append(entry)
    else:
        with open(manifest_path) as f:
            for raw in f:
                entry = _entry_from_path(raw)
                if entry is not None:
                    entries.append(entry)
    return entries


def _scan_pose_entries(pose_dir: Path, glob_pattern: str) -> list[ManifestEntry]:
    return [
        ManifestEntry(None, clip_id, segment, segment)
        for clip_id, segment in _scan_pose_dir(pose_dir, glob_pattern)
    ]


def resolve_model_and_config(spec: str) -> tuple[PoseTokenizerModel, TrainConfig, str]:
    """Load checkpoints, tolerating HF repos that only contain model config.

    Some published tokenizer repos have ``config.json``/``model.safetensors``
    but not the training ``config.yaml``. In that case we can still export as
    long as the caller supplies any needed dataset-side overrides such as
    ``--dims-csv``; otherwise ``TrainConfig`` defaults cover source/target FPS.
    """
    try:
        return _resolve_model_and_config(spec)
    except FileNotFoundError as exc:
        if not is_hf_repo_id(spec) or "config.yaml" not in str(exc):
            raise
        model = PoseTokenizerModel.from_pretrained(spec)
        return model, TrainConfig(), spec


def _count_existing_wan_chunks(
    wan_root: Path | None,
    entry: ManifestEntry,
) -> int | None:
    if wan_root is None:
        return None
    if entry.version is not None:
        folder = wan_root / entry.version / entry.clip_id / entry.segment
    else:
        folder = wan_root / "v1" / entry.clip_id / entry.segment
    if not folder.is_dir():
        return 0
    return len(list(folder.glob(f"{entry.segment}_chunk*.txt")))


class _EntryDataset(Dataset):
    """Wrap ``ChunkedPoseDataset`` while preserving manifest path metadata."""

    def __init__(self, entries: list[ManifestEntry], base: ChunkedPoseDataset):
        self.entries = entries
        self.base = base

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        entry = self.entries[idx]
        item["version"] = entry.version or ""
        item["clip_path"] = entry.clip_path
        return item


class _TokenParquetWriter:
    """Fixed-size-list parquet writer matching ``tokenize_chunks.py``."""

    def __init__(
        self,
        path: Path,
        n_codebooks: int,
        n_token_frames: int,
        chunk_frames: int,
        target_fps: int,
        checkpoint: str,
        row_group_size: int,
        zstd_level: int,
    ):
        self.K = n_codebooks
        self.N = n_token_frames
        self._kn = n_codebooks * n_token_frames
        self._row_group_size = row_group_size
        self._schema = pa.schema(
            [
                pa.field("clip_id", pa.string()),
                pa.field("chunk_path", pa.string()),
                pa.field("tokens", pa.list_(pa.int16(), self._kn)),
            ],
            metadata={
                b"n_codebooks": str(n_codebooks).encode(),
                b"n_token_frames": str(n_token_frames).encode(),
                b"chunk_frames": str(chunk_frames).encode(),
                b"target_fps": str(target_fps).encode(),
                b"checkpoint": checkpoint.encode(),
                b"tokens_layout": b"flat_KN_reshape_to_K_N",
            },
        )
        self._writer = pq.ParquetWriter(
            str(path),
            self._schema,
            compression="zstd",
            compression_level=zstd_level,
        )

    def write_batch(
        self,
        clip_ids: list[str],
        chunk_paths: list[str],
        codes: np.ndarray,
    ) -> None:
        # codes: (B, T_codes, K), int16. Flatten as K outer x N inner.
        B = codes.shape[0]
        flat = np.ascontiguousarray(codes.transpose(0, 2, 1)).reshape(B * self._kn)
        tokens_arr = pa.FixedSizeListArray.from_arrays(
            pa.array(flat, type=pa.int16()),
            self._kn,
        )
        table = pa.Table.from_arrays(
            [
                pa.array(clip_ids, type=pa.string()),
                pa.array(chunk_paths, type=pa.string()),
                tokens_arr,
            ],
            schema=self._schema,
        )
        self._writer.write_table(table, row_group_size=self._row_group_size)

    def close(self) -> None:
        self._writer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Local path to checkpoint.pt OR an HF Hub repo id like 'org/repo'",
    )
    p.add_argument(
        "--manifest",
        nargs="+",
        default=None,
        help=(
            "One or more CSV/TXT manifests. CSVs may contain clip_path rows like "
            "v2/<clip>/<segment>. If omitted, scan --pose-dir."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Root for token parquet shards.",
    )
    p.add_argument(
        "--recon-root",
        default=None,
        help=(
            "Root for per-chunk pose .safetensors. Defaults to --wan-root when "
            "set, so pose chunks are written beside existing Wan chunk files; "
            "otherwise defaults to --output-dir."
        ),
    )
    p.add_argument(
        "--pose-dir",
        default="/mnt/somfs/pose_conditioning_uncompressed",
        help="Directory holding <segment>_poses.npy files",
    )
    p.add_argument(
        "--pose-glob",
        default="*_poses.npy",
        help="Glob used when --manifest is omitted (relative to --pose-dir)",
    )
    p.add_argument(
        "--wan-root",
        default=None,
        help=(
            "Optional root of existing Wan data. v2/v3/v4 manifest rows are "
            "matched under <wan-root>/<version>/<clip>/<segment>."
        ),
    )
    p.add_argument(
        "--require-wan-chunks",
        action="store_true",
        help="Skip clips that have no matching Wan chunk files under --wan-root.",
    )
    p.add_argument(
        "--dims-csv",
        default=None,
        help="Override dims.csv from the run config (clip_stem,height,width,fps)",
    )
    p.add_argument(
        "--source-fps",
        type=int,
        default=None,
        help="Override source FPS from config.yaml/defaults.",
    )
    p.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help="Override target FPS from config.yaml/defaults.",
    )
    p.add_argument(
        "--norm-stats",
        default=None,
        help="Override norm_stats path from config.yaml/defaults.",
    )
    p.add_argument(
        "--chunk-frames",
        type=int,
        default=76,
        help="Frames per Wan generated chunk at target_fps.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Token rows buffered per parquet write.",
    )
    p.add_argument(
        "--loader-batch-size",
        type=int,
        default=1,
        help=(
            "Number of pose files per DataLoader batch. Keep at 1 to show "
            "per-file tqdm progress and avoid padding variable-length clips."
        ),
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers per process (I/O parallelism)",
    )
    p.add_argument(
        "--smooth-eurofilter",
        action="store_true",
        help="Apply One Euro smoothing to raw pose x/y coordinates before tokenization.",
    )
    p.add_argument("--euro-min-cutoff", type=float, default=3.0)
    p.add_argument("--euro-beta", type=float, default=0.3)
    p.add_argument("--euro-d-cutoff", type=float, default=1.0)
    p.add_argument(
        "--euro-min-conf",
        type=float,
        default=0.3,
        help="Confidence below which the One Euro filter holds the previous keypoint.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help=">0 processes only the first N manifest entries (debug)",
    )
    p.add_argument(
        "--row-group-size",
        type=int,
        default=10_000,
        help="Parquet row group size (chunks per group).",
    )
    p.add_argument(
        "--zstd-level",
        type=int,
        default=3,
        help="zstd compression level for parquet token shards (1-22).",
    )
    return p.parse_args()


def _safe_chunk_path(entry: ManifestEntry, chunk_idx: int) -> str:
    chunk_name = f"{entry.segment}_chunk{chunk_idx:04d}"
    if entry.version is not None:
        return f"{entry.version}/{entry.clip_id}/{chunk_name}"
    return chunk_name


def _reconstruction_path(recon_root: Path, entry: ManifestEntry, chunk_idx: int) -> Path:
    chunk_name = f"{entry.segment}_poses_chunk{chunk_idx:04d}.safetensors"
    return recon_root / entry.clip_path / chunk_name


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    rank = accelerator.process_index
    world = accelerator.num_processes
    device = accelerator.device

    model, cfg, source_name = resolve_model_and_config(args.checkpoint)
    model.to(device).eval()
    n_cb = model.config.n_codebooks
    ds_factor = model.downsampling_factor
    n_token_frames = args.chunk_frames // ds_factor
    feature_dim = model.config.input_features
    include_confidence = feature_dim == NUM_JOINTS * 3
    if feature_dim not in (NUM_JOINTS * 2, NUM_JOINTS * 3):
        raise ValueError(
            f"Unsupported model input_features={feature_dim}; "
            f"expected {NUM_JOINTS * 2} or {NUM_JOINTS * 3}."
        )
    if args.chunk_frames % ds_factor != 0:
        raise ValueError(
            f"--chunk-frames ({args.chunk_frames}) must be divisible by the model's "
            f"downsample factor ({ds_factor}). Got {args.chunk_frames % ds_factor} remainder."
        )

    if args.manifest:
        entries = [
            entry
            for manifest in args.manifest
            for entry in _parse_clip_manifest(Path(manifest))
        ]
    else:
        entries = _scan_pose_entries(Path(args.pose_dir), args.pose_glob)
    if args.limit > 0:
        entries = entries[: args.limit]

    dims_csv = args.dims_csv or cfg.dims_csv or None
    if dims_csv and not Path(dims_csv).is_file():
        cand = Path.cwd() / dims_csv
        if cand.is_file():
            dims_csv = str(cand)
    source_fps = args.source_fps or cfg.source_fps
    target_fps = args.target_fps or cfg.target_fps
    norm_stats = args.norm_stats if args.norm_stats is not None else cfg.norm_stats or None

    base_ds = ChunkedPoseDataset(
        entries=[(entry.clip_id, entry.segment) for entry in entries],
        pose_dir=args.pose_dir,
        dims_csv=dims_csv,
        source_fps=source_fps,
        target_fps=target_fps,
        chunk_frames=args.chunk_frames,
        norm_stats=norm_stats,
        smooth_eurofilter=args.smooth_eurofilter,
        euro_min_cutoff=args.euro_min_cutoff,
        euro_beta=args.euro_beta,
        euro_d_cutoff=args.euro_d_cutoff,
        euro_min_conf=args.euro_min_conf,
        include_confidence=include_confidence,
    )
    full_ds = _EntryDataset(entries, base_ds)

    rank_indices = list(range(rank, len(full_ds), world))
    rank_ds = Subset(full_ds, rank_indices)
    loader = DataLoader(
        rank_ds,
        batch_size=args.loader_batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_passthrough,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recon_root = Path(args.recon_root or args.wan_root or args.output_dir)
    recon_root.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / f"rank{rank:02d}.parquet"
    token_writer = _TokenParquetWriter(
        token_path,
        n_codebooks=n_cb,
        n_token_frames=n_token_frames,
        chunk_frames=args.chunk_frames,
        target_fps=target_fps,
        checkpoint=source_name,
        row_group_size=args.row_group_size,
        zstd_level=args.zstd_level,
    )

    if accelerator.is_main_process:
        source_kind = "HF Hub" if is_hf_repo_id(args.checkpoint) else "local"
        source_list = (
            ", ".join(args.manifest)
            if args.manifest
            else f"{args.pose_dir}/{args.pose_glob}"
        )
        accelerator.print(
            f"checkpoint:   {source_name}  ({source_kind})\n"
            f"sources:      {source_list}\n"
            f"pose-dir:     {args.pose_dir}\n"
            f"dims-csv:     {dims_csv}\n"
            f"fps:          source={source_fps}, target={target_fps}\n"
            f"norm-stats:   {norm_stats or '(not used)'}\n"
            f"wan-root:     {args.wan_root or '(not used)'}\n"
            f"token shards: {output_dir}/rank*.parquet\n"
            f"recon chunks: {recon_root}\n"
            f"smoothing:    "
            f"{'one-euro' if args.smooth_eurofilter else '(not used)'}"
            f"{f' min_cutoff={args.euro_min_cutoff} beta={args.euro_beta} d_cutoff={args.euro_d_cutoff} min_conf={args.euro_min_conf}' if args.smooth_eurofilter else ''}\n"
            f"clips:        {len(full_ds):,} total  "
            f"(~{len(rank_indices):,} per rank x {world} ranks)\n"
            f"chunk:        tokenize full clip after dropping frame 0, decode "
            f"tokens, then split tokens into {n_token_frames} steps and "
            f"reconstructions into {args.chunk_frames} frames\n"
            f"model:        {sum(p.numel() for p in model.parameters()):,} params, "
            f"downsample={ds_factor}, features={feature_dim}"
        )

    buf_codes: list[np.ndarray] = []
    buf_meta: list[tuple[str, str]] = []
    n_chunks_written = 0
    n_recon_written = 0
    n_clips_ok = 0
    n_clips_skipped = 0
    skip_reasons: dict[str, int] = {}
    wan_root = Path(args.wan_root) if args.wan_root else None

    pbar = tqdm(
        loader,
        desc=f"rank{rank} files",
        disable=not accelerator.is_local_main_process,
        total=len(loader),
        smoothing=0.05,
    )

    def flush_tokens(batch_size: int) -> None:
        nonlocal buf_codes, buf_meta, n_chunks_written
        if batch_size <= 0:
            return
        codes = np.stack(buf_codes[:batch_size], axis=0).astype(np.int16, copy=False)
        clip_ids = [m[0] for m in buf_meta[:batch_size]]
        chunk_paths = [m[1] for m in buf_meta[:batch_size]]
        token_writer.write_batch(clip_ids, chunk_paths, codes)
        n_chunks_written += batch_size
        del buf_codes[:batch_size]
        del buf_meta[:batch_size]

    for batch in pbar:
        for item in batch:
            if not item["valid"]:
                n_clips_skipped += 1
                reason = item.get("reason", "unknown")
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            entry = ManifestEntry(
                item.get("version") or None,
                item["clip_id"],
                item["segment"],
                item.get("clip_path") or item["segment"],
            )
            kp = item["keypoints"].unsqueeze(0).to(device, non_blocking=True)

            with torch.inference_mode():
                codes_list = model.tokenize(kp)
                reconstruction = model.detokenize(codes_list)

            codes = (
                torch.stack(codes_list, dim=-1)
                .squeeze(0)
                .to(torch.int16)
                .cpu()
                .numpy()
            )
            recon = reconstruction.squeeze(0).to(torch.float32).cpu()

            n_chunks = min(
                codes.shape[0] // n_token_frames,
                recon.shape[0] // args.chunk_frames,
            )
            existing_wan_chunks = _count_existing_wan_chunks(wan_root, entry)
            if existing_wan_chunks == 0:
                if args.require_wan_chunks:
                    n_clips_skipped += 1
                    skip_reasons["missing_wan_chunks"] = (
                        skip_reasons.get("missing_wan_chunks", 0) + 1
                    )
                    continue
            elif existing_wan_chunks is not None:
                n_chunks = min(n_chunks, existing_wan_chunks)

            if n_chunks == 0:
                n_clips_skipped += 1
                skip_reasons["too_few_steps_or_frames"] = (
                    skip_reasons.get("too_few_steps_or_frames", 0) + 1
                )
                continue

            n_clips_ok += 1
            for c in range(n_chunks):
                token_start = c * n_token_frames
                token_end = token_start + n_token_frames
                frame_start = c * args.chunk_frames
                frame_end = frame_start + args.chunk_frames

                chunk_path = _safe_chunk_path(entry, c)
                buf_codes.append(np.ascontiguousarray(codes[token_start:token_end]))
                buf_meta.append((entry.clip_id, chunk_path))

                recon_path = _reconstruction_path(recon_root, entry, c)
                recon_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(
                    {
                        "reconstruction": recon[frame_start:frame_end]
                        .contiguous()
                        .cpu(),
                    },
                    str(recon_path),
                    metadata={
                        "clip_id": entry.clip_id,
                        "segment": entry.segment,
                        "clip_path": entry.clip_path,
                        "chunk_path": chunk_path,
                        "chunk_frames": str(args.chunk_frames),
                        "feature_dim": str(feature_dim),
                        "target_fps": str(target_fps),
                        "checkpoint": source_name,
                        "normalized": "true",
                    },
                )
                n_recon_written += 1

        while len(buf_codes) >= args.batch_size:
            flush_tokens(args.batch_size)

        if n_chunks_written and accelerator.is_local_main_process:
            pbar.set_postfix(
                clips=n_clips_ok,
                chunks=n_chunks_written,
                recon=n_recon_written,
                skipped=n_clips_skipped,
                refresh=False,
            )

    while buf_codes:
        flush_tokens(len(buf_codes))
    token_writer.close()

    print(
        f"[rank {rank}]  wrote {n_chunks_written:,} token rows and "
        f"{n_recon_written:,} reconstruction chunks  "
        f"({n_clips_ok:,} clips ok, {n_clips_skipped:,} skipped)  "
        f"-> {token_path}"
    )
    if skip_reasons:
        print(f"[rank {rank}]  skip reasons: {skip_reasons}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shards = sorted(output_dir.glob("rank*.parquet"))
        total_bytes = sum(p.stat().st_size for p in shards)
        accelerator.print(
            f"\nDone. {len(shards)} token parquet shard(s) -> {output_dir}  "
            f"({total_bytes / 1e6:.1f} MB total)\n"
            f"Reconstruction safetensors -> {recon_root}\n"
            f"Load tokens with:\n"
            f"  import pandas as pd, glob\n"
            f"  df = pd.concat(pd.read_parquet(f) for f in "
            f"glob.glob('{output_dir}/rank*.parquet')).set_index('chunk_path')\n"
            f"  tokens = df.loc[chunk]['tokens'].reshape({n_cb}, {n_token_frames})"
        )


if __name__ == "__main__":
    main()
