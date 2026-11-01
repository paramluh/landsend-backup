"""Tokenize the full pose dataset using a trained checkpoint.

Multi-GPU via accelerate; shards the manifest across ranks and writes
column-per-codebook Parquet shards. Each row = one source pose file; the
schema has one ``tokens_cbN`` list column per codebook plus ``file_id``,
``n_frames``, and ``n_code_steps``. Skipped clips (tiny shoulder width,
unreadable, malformed) are omitted from output and counted in the rank
summary.

Output layout::

    <output_dir>/
        rank00_shard00000.parquet
        rank00_shard00001.parquet
        ...
        rank03_shard00012.parquet

Load later as a HuggingFace dataset with zero conversion::

    from datasets import load_dataset
    ds = load_dataset("parquet", data_files="<output_dir>/*.parquet", split="train")

The training YAML is auto-resolved from the model's run directory and is
required to know data paths / fps / normalisation. ``--checkpoint`` accepts
either a local path or an HF Hub repo ID:

    Local: ``checkpoints/wide-l2norm/step_100000/checkpoint.pt``
        → config from ``checkpoints/wide-l2norm/config.yaml``
        → weights via ``torch.load`` + ``load_state_dict``.

    HF Hub: ``org/pose-tokenizer-wide-l2norm``
        → model via ``PoseTokenizerModel.from_pretrained(repo_id)``
          (uses the ``PyTorchModelHubMixin`` already on the class)
        → config via ``hf_hub_download(repo_id, "config.yaml")``.
        → To upload, ``model.push_to_hub(...)`` plus a separate
          ``HfApi().upload_file(path_or_fileobj=<run>/config.yaml, …)``.

Defaults to ``--batch-size 1`` for boundary-perfect tokens (no zero-pad
contamination near clip ends). Raise it for length-uniform clip sets if
you need higher GPU throughput.

Usage::

    accelerate launch --multi_gpu --num_processes=4 \\
        scripts/tokenize_dataset.py \\
        --checkpoint checkpoints/wide-l2norm/step_100000/checkpoint.pt \\
        --output-dir tokens/wide-l2norm/
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from pose_tokenizer import PoseTokenizerModel
from pose_tokenizer.config import PoseTokenizerConfig, TrainConfig
from pose_tokenizer.data.dataset import (
    _append_confidence_features,
    _resample_clip,
    load_clip,
)
from pose_tokenizer.data.kinematic import NUM_JOINTS


# ---------------------------------------------------------------------------
# Dataset: clip-level, 1:1 with manifest (no skip-retry like training).
# ---------------------------------------------------------------------------

class TokenizeDataset(Dataset):
    """Iterates the manifest 1:1 — failing clips return ``valid=False``."""

    def __init__(
        self,
        data_dir: str | Path,
        manifest: str | Path,
        dims_csv: str | Path | None,
        source_fps: int,
        target_fps: int,
        norm_stats: str | Path | None = None,
        include_confidence: bool = False,
    ):
        self.data_dir = Path(data_dir)
        with open(manifest) as f:
            self.files = [self.data_dir / line.strip() for line in f if line.strip()]

        self.dims: dict[str, tuple[float, float]] = {}
        if dims_csv:
            with open(dims_csv) as f:
                for row in csv.DictReader(f):
                    self.dims[row["clip_stem"]] = (
                        float(row["height"]),
                        float(row["width"]),
                    )

        self.source_fps = source_fps
        self.target_fps = target_fps
        self.include_confidence = include_confidence
        self.feature_dim = NUM_JOINTS * (3 if include_confidence else 2)
        self.norm_center: np.ndarray | None = None
        self.norm_scale: np.ndarray | None = None
        if norm_stats and os.path.isfile(norm_stats):
            st = np.load(norm_stats)
            self.norm_center = st["center"].astype(np.float32)
            self.norm_scale = st["scale"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        stem = path.stem
        if stem.endswith("_poses"):
            stem = stem[: -len("_poses")]
        orig_h, orig_w = self.dims.get(stem, (None, None))
        rel = str(path.relative_to(self.data_dir))

        clip = load_clip(
            path, orig_h=orig_h, orig_w=orig_w,
            norm_center=self.norm_center, norm_scale=self.norm_scale,
        )
        if clip is not None and self.source_fps != self.target_fps:
            clip = _resample_clip(clip, self.source_fps, self.target_fps)
        if clip is None:
            return {
                "valid": False,
                "file_id": rel,
                "keypoints": torch.zeros(0, self.feature_dim),
                "n_frames": 0,
            }

        kp_np = (
            _append_confidence_features(clip["keypoints"], clip["confidence"])
            if self.include_confidence else clip["keypoints"]
        )
        kp = torch.from_numpy(kp_np).float()
        return {"valid": True, "file_id": rel, "keypoints": kp, "n_frames": int(kp.shape[0])}


def collate(batch: list[dict]) -> dict:
    """Pad keypoints to the max clip length in the batch."""
    max_t = max((item["n_frames"] for item in batch), default=0) or 1
    feature_dim = batch[0]["keypoints"].shape[-1]
    kp = torch.zeros(len(batch), max_t, feature_dim, dtype=torch.float32)
    for i, item in enumerate(batch):
        t = item["n_frames"]
        if t > 0:
            kp[i, :t] = item["keypoints"]
    return {
        "valid": torch.tensor([item["valid"] for item in batch], dtype=torch.bool),
        "file_id": [item["file_id"] for item in batch],
        "keypoints": kp,
        "n_frames": torch.tensor([item["n_frames"] for item in batch], dtype=torch.int32),
    }


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------

def build_schema(n_codebooks: int) -> pa.Schema:
    fields = [
        pa.field("file_id", pa.string()),
        pa.field("n_frames", pa.int32()),
        pa.field("n_code_steps", pa.int32()),
    ]
    for i in range(n_codebooks):
        fields.append(pa.field(f"tokens_cb{i}", pa.list_(pa.int16())))
    return pa.schema(fields)


def flush_shard(buffer: list[dict], schema: pa.Schema, output_dir: Path, rank: int, shard_idx: int) -> None:
    if not buffer:
        return
    path = output_dir / f"rank{rank:02d}_shard{shard_idx:05d}.parquet"
    pq.write_table(
        pa.Table.from_pylist(buffer, schema=schema),
        str(path), compression="zstd", compression_level=3,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--checkpoint", required=True,
        help="Local path to checkpoint.pt OR an HF Hub repo ID like 'org/repo'",
    )
    p.add_argument("--output-dir", required=True, help="Where to write rank*_shard*.parquet")
    p.add_argument("--manifest", default=None, help="Override the manifest path from the resolved config")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Per-GPU batch size. 1 = boundary-perfect tokens (no zero-pad bleed)")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per process")
    p.add_argument("--shard-size", type=int, default=5000, help="Rows per Parquet shard")
    p.add_argument("--limit", type=int, default=0, help=">0 → tokenize only the first N files (debug)")
    return p.parse_args()


# HF repo IDs look like ``namespace/repo`` (single slash, alnum + ``.-_``).
# We use this regex + a path-prefix exclusion list to avoid swallowing a
# mistyped local path into the HF code path (which would produce a cryptic
# HFValidationError instead of a clean "local file not found").
_HF_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
_CKPT_EXTS = (".pt", ".bin", ".safetensors", ".pth", ".yaml", ".json")


def is_hf_repo_id(spec: str) -> bool:
    """Heuristically classify ``spec`` as a local path vs an HF Hub repo ID.

    Returns False (= treat as local) if the spec already resolves to a file
    on disk, begins with a path-like prefix, or ends with a checkpoint
    extension. Otherwise: must match ``namespace/repo``.
    """
    if Path(spec).is_file():
        return False
    if spec.startswith(("/", "./", "../")):
        return False
    if spec.endswith(_CKPT_EXTS):
        return False
    return bool(_HF_REPO_PATTERN.match(spec))


def resolve_model_and_config(spec: str) -> tuple["PoseTokenizerModel", "TrainConfig", str]:
    """Return (model_with_weights_loaded, train_cfg, display_name).

    ``spec`` is either a local checkpoint.pt path or an HF Hub repo ID. In
    both cases the training YAML is the ``config.yaml`` that lives next to
    the model artifact (run dir locally, repo root on the Hub).
    """
    if is_hf_repo_id(spec):
        from huggingface_hub import hf_hub_download
        model = PoseTokenizerModel.from_pretrained(spec)
        try:
            cfg_path = hf_hub_download(repo_id=spec, filename="config.yaml")
        except Exception as e:
            raise FileNotFoundError(
                f"HF repo '{spec}' is missing 'config.yaml' at its root. "
                f"Upload the training YAML alongside the model: "
                f"`HfApi().upload_file(path_or_fileobj=<run>/config.yaml, "
                f"path_in_repo='config.yaml', repo_id='{spec}')`."
            ) from e
        cfg, _ = TrainConfig.from_yaml(cfg_path)
        return model, cfg, spec

    spec_path = Path(spec)
    ckpt_dir = spec_path.parent if spec_path.is_file() else spec_path
    if ckpt_dir.is_dir() and (
        (ckpt_dir / "config.json").is_file() or (ckpt_dir / "model.safetensors").is_file()
    ):
        cfg_path = ckpt_dir / "config.yaml"
        if not cfg_path.is_file():
            cfg_path = ckpt_dir.parent / "config.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"Could not find training config for {spec}. Expected config.yaml "
                "in the checkpoint dir or its parent run dir."
            )
        cfg, _ = TrainConfig.from_yaml(cfg_path)
        model = PoseTokenizerModel.from_pretrained(str(ckpt_dir))
        return model, cfg, str(ckpt_dir)

    if not spec_path.is_file():
        raise FileNotFoundError(
            f"Local checkpoint not found: {spec}\n"
            f"(Also doesn't look like an HF repo ID — expected 'namespace/repo'.)"
        )
    cfg_path = spec_path.resolve().parent.parent / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Could not find training config at {cfg_path}. "
            f"Expected layout: <run_dir>/step_N/checkpoint.pt + <run_dir>/config.yaml."
        )
    cfg, model_overrides = TrainConfig.from_yaml(cfg_path)
    model = (
        PoseTokenizerModel(config=PoseTokenizerConfig(**model_overrides))
        if model_overrides else PoseTokenizerModel()
    )
    ckpt = torch.load(spec, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model, cfg, spec


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    rank = accelerator.process_index
    world = accelerator.num_processes

    # ── Resolve model + training config (local path OR HF repo ID) ─────────
    model, cfg, source_name = resolve_model_and_config(args.checkpoint)
    model.to(accelerator.device).eval()
    n_cb = model.config.n_codebooks
    ds_factor = model.downsampling_factor
    include_confidence = model.config.input_features == NUM_JOINTS * 3
    if model.config.input_features not in (NUM_JOINTS * 2, NUM_JOINTS * 3):
        raise ValueError(
            f"Unsupported model input_features={model.config.input_features}; "
            f"expected {NUM_JOINTS * 2} or {NUM_JOINTS * 3}."
        )

    # ── Dataset: shard manifest across ranks (strided, no DistSampler) ─────
    full_ds = TokenizeDataset(
        data_dir=cfg.data_dir,
        manifest=args.manifest or cfg.manifest,
        dims_csv=cfg.dims_csv or None,
        source_fps=cfg.source_fps, target_fps=cfg.target_fps,
        norm_stats=cfg.norm_stats or None,
        include_confidence=include_confidence,
    )
    if args.limit > 0:
        full_ds.files = full_ds.files[: args.limit]

    rank_indices = list(range(rank, len(full_ds), world))
    rank_ds = Subset(full_ds, rank_indices)
    loader = DataLoader(
        rank_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    if accelerator.is_main_process:
        source_kind = "HF Hub" if is_hf_repo_id(args.checkpoint) else "local"
        accelerator.print(
            f"checkpoint: {source_name}  ({source_kind})\n"
            f"manifest:   {args.manifest or cfg.manifest}\n"
            f"files:      {len(full_ds):,} total  (~{len(rank_indices):,} per rank × {world} ranks)\n"
            f"model:      {sum(p.numel() for p in model.parameters()):,} params, "
            f"{n_cb} codebooks, downsample={ds_factor}, "
            f"features={model.config.input_features}"
        )

    # ── Output directory + schema ──────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = build_schema(n_cb)

    # ── Tokenize loop ──────────────────────────────────────────────────────
    buffer: list[dict] = []
    shard_idx = 0
    n_written = 0
    n_skipped = 0
    pbar = tqdm(
        loader, desc=f"rank{rank}", disable=not accelerator.is_local_main_process,
        total=len(loader), smoothing=0.05,
    )

    with torch.inference_mode():
        for batch in pbar:
            kp = batch["keypoints"].to(accelerator.device, non_blocking=True)
            # tokenize → list[Tensor(B, T_codes)] of length n_cb
            codes_list = model.tokenize(kp)
            # Stack to (B, T_codes, n_cb), cast to int16 once, move to host once
            codes = torch.stack(codes_list, dim=-1).to(torch.int16).cpu().numpy()

            for i in range(kp.shape[0]):
                if not bool(batch["valid"][i]):
                    n_skipped += 1
                    continue
                t_frames = int(batch["n_frames"][i])
                # ceil(T / downsample) — same as the encoder's output length
                # given strided convs with default same-padding.
                t_codes = min((t_frames + ds_factor - 1) // ds_factor, codes.shape[1])
                row = {
                    "file_id": batch["file_id"][i],
                    "n_frames": t_frames,
                    "n_code_steps": t_codes,
                }
                for c in range(n_cb):
                    row[f"tokens_cb{c}"] = codes[i, :t_codes, c].tolist()
                buffer.append(row)

            if len(buffer) >= args.shard_size:
                flush_shard(buffer, schema, output_dir, rank, shard_idx)
                n_written += len(buffer)
                buffer.clear()
                shard_idx += 1

    if buffer:
        flush_shard(buffer, schema, output_dir, rank, shard_idx)
        n_written += len(buffer)
        shard_idx += 1
        buffer.clear()

    print(f"[rank {rank}]  wrote {n_written:,} rows in {shard_idx} shards  "
          f"({n_skipped:,} skipped)")

    # ── Summary on rank 0 ──────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shards = sorted(output_dir.glob("rank*_shard*.parquet"))
        total_bytes = sum(p.stat().st_size for p in shards)
        accelerator.print(
            f"\nDone. {len(shards)} parquet shard(s) → {output_dir}  "
            f"({total_bytes / 1e6:.1f} MB)\n"
            f"Load with:\n"
            f"  from datasets import load_dataset\n"
            f"  ds = load_dataset('parquet', "
            f"data_files='{output_dir}/*.parquet', split='train')"
        )


if __name__ == "__main__":
    main()
