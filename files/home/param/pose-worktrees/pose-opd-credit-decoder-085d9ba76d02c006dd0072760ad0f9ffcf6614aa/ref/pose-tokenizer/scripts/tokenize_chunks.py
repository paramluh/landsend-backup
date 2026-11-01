"""Tokenize pose .npy files into fixed-size wan22-aligned chunks.

Each manifest row points at a clip *directory* (the same layout used by the
wan22 video pipeline)::

    v1/<video_id>/<segment>/

Per segment we:

1. Locate the pose file at ``<pose-dir>/<segment>_poses.npy``.
2. Apply the same kinematic-chain processing the model was trained with
   (``pose_tokenizer.data.dataset.load_clip``).
3. Resample to the model's ``target_fps`` (from the run's ``config.yaml``).
4. Skip the very first frame — by convention the first frame of every wan22
   77-frame video chunk is the conditioning frame shared with the previous
   chunk's last frame, so the 76 frames that follow are what the pose tokens
   should describe.
5. Encode the *entire* remaining pose sequence in one model pass, preserving
   temporal context across video chunk boundaries.
6. Carve the resulting token timeline into non-overlapping chunks of
   ``chunk_frames / downsample_factor`` token steps (default 76 / 4 = 19).
   The partial token tail is dropped.

Output is one Parquet file per rank, one row per chunk (default;
``--format csv`` falls back to the text format used in v1):

    clip_id     string                     — video id, e.g. "7601221154807794975"
    chunk_path  string                     — "<segment>_chunk{idx:04d}"
                                              (matches the wan22 caption
                                              file naming, so rows join to
                                              latents / text by filename).
    tokens      fixed_size_list<int16>[K·N] — K codebooks × N codes per chunk,
                                              stored *flat* (reshape on load).

The Parquet schema also carries the relevant dims (``n_codebooks``,
``n_token_frames``, ``chunk_frames``, ``target_fps``, ``checkpoint``) in the
file-level metadata so a loader can verify / reshape automatically.

Why fixed_size_list<int16> instead of CSV/JSON: at 8 codebooks × 19 token
frames per chunk we pay 304 bytes/chunk of raw tokens (vs ~600-800 in
JSON-as-text). With zstd compression of the int16 column on top, expect
roughly **4–10× smaller** files than the CSV variant at 1.5M segments.

Loading example::

    import pandas as pd, pyarrow.parquet as pq, glob

    # Whole dataset into pandas, indexed by chunk_path (like the CSV path)
    df = pd.concat(
        pd.read_parquet(f) for f in glob.glob("tokens/vansh-chunks/*.parquet")
    ).set_index("chunk_path")

    # Read schema metadata to know the (K, N) shape (avoids hard-coding)
    meta = pq.read_metadata("tokens/vansh-chunks/rank00.parquet").metadata
    K = int(meta[b"n_codebooks"]); N = int(meta[b"n_token_frames"])

    tokens = df.loc["7597427265118555405_..._chunk0001"]["tokens"].reshape(K, N)

Each source clip is encoded as one full sequence (no per-window model
forwards). ``--batch-size`` controls how many token rows are buffered before
writing to disk, not the model forward batch size.

Usage::

    # Single GPU, smoke test over explicit vansh-style manifest
    python scripts/tokenize_chunks.py \\
        --checkpoint checkpoints/wide-l2norm/step_55000/checkpoint.pt \\
        --manifest vansh_files.txt \\
        --output-dir tokens/vansh-chunks/ \\
        --limit 16

    # Scan every available pose file and only emit rows with matching Wan chunks
    python scripts/tokenize_chunks.py \\
        --checkpoint checkpoints/wide-l2norm/step_55000/checkpoint.pt \\
        --pose-dir /mnt/somfs/pose_conditioning_uncompressed \\
        --wan-root /mnt/somfs/hands \\
        --require-wan-chunks \\
        --dims-csv dims.csv \\
        --output-dir tokens/all-vansh-chunks/

    # All 8 H100s
    accelerate launch --multi_gpu --num_processes=8 \\
        scripts/tokenize_chunks.py \\
        --checkpoint checkpoints/wide-l2norm/step_55000/checkpoint.pt \\
        --manifest vansh_files.txt \\
        --output-dir tokens/vansh-chunks/ \\
        --batch-size 128
"""

from __future__ import annotations

import argparse
import csv
import json
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
    _process_poses_to_clip,
    _resample_clip,
    load_clip,
)
from pose_tokenizer.data.kinematic import NUM_JOINTS


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

def _parse_manifest(manifest_path: Path) -> list[tuple[str, str]]:
    """Read a manifest → list of ``(clip_id, segment)`` pairs.

    Supported line formats:
    - ``v1/<clip_id>/<segment>/`` (vansh/Wan-style)
    - ``<segment>_poses.npy`` / ``<segment>.npy`` (plain ``ls`` output)

    Header rows like ``"paths"`` are silently skipped.
    """
    entries: list[tuple[str, str]] = []
    with open(manifest_path) as f:
        for raw in f:
            line = raw.strip().rstrip("/")
            if not line or line == "paths":
                continue
            name = Path(line).name
            if name.endswith((".npy", ".npz")):
                segment = re.sub(r"(_poses)?\.(npy|npz)$", "", name)
                clip_id = segment.split("_", 1)[0]
                entries.append((clip_id, segment))
                continue
            parts = line.split("/")
            # Expect [..., <clip_id>, <segment>]; tolerate longer prefixes.
            if len(parts) < 2:
                continue
            clip_id = parts[-2]
            segment = parts[-1]
            entries.append((clip_id, segment))
    return entries


def _scan_pose_dir(pose_dir: Path, glob_pattern: str) -> list[tuple[str, str]]:
    """Scan ``pose_dir`` for dense pose files."""
    entries: list[tuple[str, str]] = []
    for path in sorted(pose_dir.glob(glob_pattern)):
        name = path.name
        if not name.endswith((".npy", ".npz")):
            continue
        segment = re.sub(r"(_poses)?\.(npy|npz)$", "", name)
        clip_id = segment.split("_", 1)[0]
        entries.append((clip_id, segment))
    if not entries:
        raise FileNotFoundError(
            f"No pose files matching {glob_pattern!r} found under {pose_dir}"
        )
    return entries


def _count_existing_wan_chunks(
    wan_root: Path | None,
    clip_id: str,
    segment: str,
) -> int | None:
    """Count existing ``<segment>_chunkXXXX.txt`` files in the Wan layout."""
    if wan_root is None:
        return None
    folder = wan_root / "v1" / clip_id / segment
    if not folder.is_dir():
        return 0
    return len(list(folder.glob(f"{segment}_chunk*.txt")))


# ---------------------------------------------------------------------------
# Dataset: one item = one source pose file, returning the full post-skip clip
# ---------------------------------------------------------------------------

class _OneEuroFilter:
    """Vectorized One Euro filter for one keypoint coordinate pair."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x.astype(np.float32, copy=True)
            self.dx_prev = np.zeros_like(x, dtype=np.float32)
            return self.x_prev.copy()

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat.astype(np.float32, copy=False)
        self.dx_prev = dx_hat.astype(np.float32, copy=False)
        return self.x_prev.copy()


def _smooth_poses_one_euro(
    poses: np.ndarray,
    fps: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
    min_conf: float,
) -> np.ndarray:
    """Smooth raw pose x/y tracks, freezing low-confidence observations."""
    if poses.shape[0] <= 1:
        return poses

    dt = 1.0 / float(fps)
    smoothed = poses.astype(np.float32, copy=True)
    filters = [
        _OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        for _ in range(poses.shape[1])
    ]

    for t in range(poses.shape[0]):
        for j, filt in enumerate(filters):
            xy = poses[t, j, :2].astype(np.float32, copy=False)
            conf = float(poses[t, j, 2])
            if conf >= min_conf or filt.x_prev is None:
                smoothed[t, j, :2] = filt(xy, dt)
            else:
                smoothed[t, j, :2] = filt.x_prev
    return smoothed


def _load_clip_with_optional_smoothing(
    pose_path: Path,
    orig_h: float | None,
    orig_w: float | None,
    norm_center: np.ndarray | None,
    norm_scale: np.ndarray | None,
    smooth_eurofilter: bool,
    smoothing_fps: float,
    euro_min_cutoff: float,
    euro_beta: float,
    euro_d_cutoff: float,
    euro_min_conf: float,
) -> dict[str, np.ndarray] | None:
    if not smooth_eurofilter:
        return load_clip(
            pose_path,
            orig_h=orig_h,
            orig_w=orig_w,
            norm_center=norm_center,
            norm_scale=norm_scale,
        )

    if orig_h is None or orig_w is None:
        return None
    try:
        poses = np.asarray(np.load(pose_path, mmap_mode="r"), dtype=np.float32)
    except FileNotFoundError:
        raise
    except Exception:
        return None
    if poses.ndim != 3 or poses.shape[1:] != (133, 3):
        return None

    poses = _smooth_poses_one_euro(
        poses,
        fps=smoothing_fps,
        min_cutoff=euro_min_cutoff,
        beta=euro_beta,
        d_cutoff=euro_d_cutoff,
        min_conf=euro_min_conf,
    )
    return _process_poses_to_clip(
        poses,
        float(orig_h),
        float(orig_w),
        norm_center,
        norm_scale,
    )


class ChunkedPoseDataset(Dataset):
    """Per-clip loader. ``__getitem__`` returns one full post-skip pose sequence.

    ``valid=False`` means the file was unreadable / dropped by ``load_clip``
    (tiny shoulder width, missing file, …) or had no frames after resampling +
    first-frame skip.
    """

    def __init__(
        self,
        entries: list[tuple[str, str]],
        pose_dir: str | Path,
        dims_csv: str | Path | None,
        source_fps: int,
        target_fps: int,
        chunk_frames: int,
        norm_stats: str | Path | None = None,
        smooth_eurofilter: bool = False,
        euro_min_cutoff: float = 3.0,
        euro_beta: float = 0.3,
        euro_d_cutoff: float = 1.0,
        euro_min_conf: float = 0.3,
        include_confidence: bool = False,
    ):
        self.entries = entries
        self.pose_dir = Path(pose_dir)
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.chunk_frames = chunk_frames
        self.smooth_eurofilter = smooth_eurofilter
        self.euro_min_cutoff = euro_min_cutoff
        self.euro_beta = euro_beta
        self.euro_d_cutoff = euro_d_cutoff
        self.euro_min_conf = euro_min_conf
        self.include_confidence = include_confidence
        self.feature_dim = NUM_JOINTS * (3 if include_confidence else 2)

        self.dims: dict[str, tuple[float, float]] = {}
        self.fps_by_segment: dict[str, float] = {}
        if dims_csv:
            with open(dims_csv) as f:
                for row in csv.DictReader(f):
                    self.dims[row["clip_stem"]] = (
                        float(row["height"]),
                        float(row["width"]),
                    )
                    if row.get("fps"):
                        self.fps_by_segment[row["clip_stem"]] = float(row["fps"])

        self.norm_center: np.ndarray | None = None
        self.norm_scale: np.ndarray | None = None
        if norm_stats and os.path.isfile(norm_stats):
            st = np.load(norm_stats)
            self.norm_center = st["center"].astype(np.float32)
            self.norm_scale = st["scale"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.entries)

    def _empty(self, clip_id: str, segment: str, reason: str) -> dict:
        return {
            "valid": False,
            "clip_id": clip_id,
            "segment": segment,
            "reason": reason,
            "keypoints": torch.zeros(0, self.feature_dim, dtype=torch.float32),
            "n_frames": 0,
        }

    def _pose_path(self, segment: str) -> Path | None:
        for name in (
            f"{segment}_poses.npy",
            f"{segment}.npy",
            f"{segment}_poses.npz",
            f"{segment}.npz",
        ):
            path = self.pose_dir / name
            if path.is_file():
                return path
        return None

    def __getitem__(self, idx: int) -> dict:
        clip_id, segment = self.entries[idx]
        pose_path = self._pose_path(segment)
        if pose_path is None:
            return self._empty(clip_id, segment, "missing_pose_file")

        orig_h, orig_w = self.dims.get(segment, (None, None))
        source_fps = self.fps_by_segment.get(segment, self.source_fps)
        try:
            clip = _load_clip_with_optional_smoothing(
                pose_path,
                orig_h=orig_h, orig_w=orig_w,
                norm_center=self.norm_center, norm_scale=self.norm_scale,
                smooth_eurofilter=self.smooth_eurofilter,
                smoothing_fps=source_fps,
                euro_min_cutoff=self.euro_min_cutoff,
                euro_beta=self.euro_beta,
                euro_d_cutoff=self.euro_d_cutoff,
                euro_min_conf=self.euro_min_conf,
            )
        except FileNotFoundError:
            return self._empty(clip_id, segment, "missing_pose_file")
        if clip is None:
            return self._empty(clip_id, segment, "load_clip_returned_none")

        if source_fps != self.target_fps:
            clip = _resample_clip(clip, source_fps, self.target_fps)

        kp = clip["keypoints"]  # (T, 110) float32, target_fps frames
        conf = clip["confidence"]  # (T, 55) float32, target_fps frames
        gs = clip["global_state"]  # (T, 3) float32: root_y, root_x, shoulder_width
        # Skip the first frame (conditioning frame for wan22 chunk 0).
        kp = kp[1:]
        conf = conf[1:]
        gs = gs[1:]
        if kp.shape[0] == 0:
            return self._empty(clip_id, segment, "too_short")
        if self.include_confidence:
            kp = _append_confidence_features(kp, conf)

        return {
            "valid": True,
            "clip_id": clip_id,
            "segment": segment,
            "keypoints": torch.from_numpy(np.ascontiguousarray(kp)).float(),
            "global_state": torch.from_numpy(np.ascontiguousarray(gs)).float(),
            "n_frames": int(kp.shape[0]),
        }


def collate_passthrough(batch: list[dict]) -> list[dict]:
    """Don't stack — chunks-per-clip varies. Cross-clip uniform batching is
    handled by the chunk accumulator in :func:`main`."""
    return batch


# ---------------------------------------------------------------------------
# Model + config resolution (local checkpoint OR HF Hub repo id)
# ---------------------------------------------------------------------------

_HF_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
_CKPT_EXTS = (".pt", ".bin", ".safetensors", ".pth", ".yaml", ".json")


def is_hf_repo_id(spec: str) -> bool:
    if Path(spec).is_file():
        return False
    if spec.startswith(("/", "./", "../")):
        return False
    if spec.endswith(_CKPT_EXTS):
        return False
    return bool(_HF_REPO_PATTERN.match(spec))


def resolve_model_and_config(spec: str) -> tuple[PoseTokenizerModel, TrainConfig, str]:
    """Mirror of ``tokenize_dataset.resolve_model_and_config`` — accepts a
    local checkpoint path or an HF Hub repo id. The training YAML is the
    ``config.yaml`` next to the model artefact (run-dir locally, repo root
    on the Hub).
    """
    if is_hf_repo_id(spec):
        from huggingface_hub import hf_hub_download
        model = PoseTokenizerModel.from_pretrained(spec)
        try:
            cfg_path = hf_hub_download(repo_id=spec, filename="config.yaml")
        except Exception as e:
            raise FileNotFoundError(
                f"HF repo '{spec}' is missing 'config.yaml' at its root."
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
            f"(Also doesn't look like an HF repo id — expected 'namespace/repo'.)"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--checkpoint", required=True,
        help="Local path to checkpoint.pt OR an HF Hub repo id like 'org/repo'",
    )
    p.add_argument(
        "--manifest", default=None,
        help=(
            "Optional vansh-style txt: one v1/<clip>/<segment>/ per line. "
            "If omitted, scan --pose-dir for *_poses.npy files."
        ),
    )
    p.add_argument("--output-dir", required=True,
                   help="Where to write rank{R:02d}.csv")
    p.add_argument(
        "--pose-dir",
        default="/mnt/somfs/pose_conditioning_uncompressed",
        help="Directory holding <segment>_poses.npy files",
    )
    p.add_argument(
        "--pose-glob", default="*_poses.npy",
        help="Glob used when --manifest is omitted (relative to --pose-dir)",
    )
    p.add_argument(
        "--wan-root", default=None,
        help=(
            "Optional root of existing Wan data, e.g. /mnt/somfs/hands. "
            "When set, output chunks are capped to existing "
            "v1/<clip>/<segment>/<segment>_chunkXXXX.txt files."
        ),
    )
    p.add_argument(
        "--require-wan-chunks", action="store_true",
        help="Skip clips that have no matching Wan chunk files under --wan-root.",
    )
    p.add_argument(
        "--dims-csv", default=None,
        help="Override dims.csv from the run config (clip_stem,height,width,fps)",
    )
    p.add_argument("--chunk-frames", type=int, default=76,
                   help="Frames per chunk at target_fps (matches wan22 generated portion)")
    p.add_argument(
        "--batch-size", type=int, default=128,
        help=(
            "Token rows buffered per write. Kept for backward compatibility; "
            "model forward is always one full clip at a time."
        ),
    )
    p.add_argument(
        "--loader-batch-size", type=int, default=1,
        help=(
            "Number of pose files per DataLoader batch. Keep at 1 to show "
            "per-file tqdm progress and avoid padding variable-length clips."
        ),
    )
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader workers per process (I/O parallelism)")
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
    p.add_argument("--limit", type=int, default=0,
                   help=">0 → process only the first N manifest entries (debug)")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                   help="Output format. Parquet (int16, zstd) is ~4-10× smaller "
                        "than CSV/JSON and faster to load; CSV is human-readable.")
    p.add_argument("--row-group-size", type=int, default=10_000,
                   help="Parquet row group size (chunks per group). Larger = "
                        "better compression; smaller = faster random access.")
    p.add_argument("--zstd-level", type=int, default=3,
                   help="zstd compression level for the parquet output (1-22).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Writers: thin abstraction so the inference loop is format-agnostic.
# ---------------------------------------------------------------------------

class _CsvWriter:
    """Header + JSON-encoded ``[K, N]`` per row. Same on-disk shape as v1."""

    def __init__(self, path: Path):
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["clip_id", "chunk_path", "tokens"])

    def write_batch(self, clip_ids: list[str], chunk_paths: list[str],
                    codes: np.ndarray) -> None:
        # codes: (B, T_codes, K) int16. Transpose so json is K outer × N inner.
        for i in range(codes.shape[0]):
            self._w.writerow([
                clip_ids[i], chunk_paths[i],
                json.dumps(codes[i].T.tolist(), separators=(",", ":")),
            ])

    def close(self) -> None:
        self._f.close()


class _ParquetWriter:
    """Fixed-size-list parquet writer (tokens stored flat, shape K·N).

    Carries (K, N, chunk_frames, target_fps, checkpoint) in file-level
    metadata so a downstream loader can reshape without hard-coding.
    """

    def __init__(self, path: Path, n_codebooks: int, n_token_frames: int,
                 chunk_frames: int, target_fps: int, checkpoint: str,
                 row_group_size: int, zstd_level: int):
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

    def write_batch(self, clip_ids: list[str], chunk_paths: list[str],
                    codes: np.ndarray) -> None:
        # codes: (B, T_codes, K) int16 — flatten each row as K outer × N inner
        # (transpose then ravel) so the on-disk order matches a (K, N) reshape.
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
        # Cap row-group size for better random-access locality.
        self._writer.write_table(table, row_group_size=self._row_group_size)

    def close(self) -> None:
        self._writer.close()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    rank = accelerator.process_index
    world = accelerator.num_processes
    device = accelerator.device

    # ── Resolve model + training config (local path OR HF repo id) ─────────
    model, cfg, source_name = resolve_model_and_config(args.checkpoint)
    model.to(device).eval()
    n_cb = model.config.n_codebooks
    ds_factor = model.downsampling_factor
    n_token_frames = args.chunk_frames // ds_factor
    include_confidence = model.config.input_features == NUM_JOINTS * 3
    if model.config.input_features not in (NUM_JOINTS * 2, NUM_JOINTS * 3):
        raise ValueError(
            f"Unsupported model input_features={model.config.input_features}; "
            f"expected {NUM_JOINTS * 2} or {NUM_JOINTS * 3}."
        )

    if args.chunk_frames % ds_factor != 0:
        raise ValueError(
            f"--chunk-frames ({args.chunk_frames}) must be divisible by the model's "
            f"downsample factor ({ds_factor}). Got {args.chunk_frames % ds_factor} remainder."
        )

    # ── Build the source clip list ─────────────────────────────────────────
    if args.manifest:
        entries = _parse_manifest(Path(args.manifest))
    else:
        entries = _scan_pose_dir(Path(args.pose_dir), args.pose_glob)
    if args.limit > 0:
        entries = entries[: args.limit]

    dims_csv = args.dims_csv or cfg.dims_csv or None
    if dims_csv and not Path(dims_csv).is_file():
        # Resolve relative to repo root if user passed a bare filename like "dims.csv".
        cand = Path.cwd() / dims_csv
        if cand.is_file():
            dims_csv = str(cand)

    full_ds = ChunkedPoseDataset(
        entries=entries,
        pose_dir=args.pose_dir,
        dims_csv=dims_csv,
        source_fps=cfg.source_fps,
        target_fps=cfg.target_fps,
        chunk_frames=args.chunk_frames,
        norm_stats=cfg.norm_stats or None,
        smooth_eurofilter=args.smooth_eurofilter,
        euro_min_cutoff=args.euro_min_cutoff,
        euro_beta=args.euro_beta,
        euro_d_cutoff=args.euro_d_cutoff,
        euro_min_conf=args.euro_min_conf,
        include_confidence=include_confidence,
    )

    # ── Shard the manifest across ranks (strided) ──────────────────────────
    rank_indices = list(range(rank, len(full_ds), world))
    rank_ds = Subset(full_ds, rank_indices)
    loader = DataLoader(
        rank_ds,
        # Clips have variable lengths and must be encoded as full sequences.
        # Keep this at 1 by default so every tqdm update corresponds to one
        # pose file and no padding/batched model forward is implied.
        batch_size=args.loader_batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_passthrough,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )

    if accelerator.is_main_process:
        source_kind = "HF Hub" if is_hf_repo_id(args.checkpoint) else "local"
        source_list = args.manifest or f"{args.pose_dir}/{args.pose_glob}"
        accelerator.print(
            f"checkpoint:   {source_name}  ({source_kind})\n"
            f"sources:      {source_list}\n"
            f"pose-dir:     {args.pose_dir}\n"
            f"dims-csv:     {dims_csv}\n"
            f"wan-root:     {args.wan_root or '(not used)'}\n"
            f"smoothing:    "
            f"{'one-euro' if args.smooth_eurofilter else '(not used)'}"
            f"{f' min_cutoff={args.euro_min_cutoff} beta={args.euro_beta} d_cutoff={args.euro_d_cutoff} min_conf={args.euro_min_conf}' if args.smooth_eurofilter else ''}\n"
            f"clips:        {len(full_ds):,} total  (~{len(rank_indices):,} per rank × {world} ranks)\n"
            f"chunk:        encode full clip after dropping frame 0, then split "
            f"tokens into {n_token_frames} steps × {n_cb} codebooks\n"
            f"model:        {sum(p.numel() for p in model.parameters()):,} params, "
            f"downsample={ds_factor}, features={model.config.input_features}"
        )

    # ── Output: one shard per rank ─────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = "parquet" if args.format == "parquet" else "csv"
    out_path = output_dir / f"rank{rank:02d}.{ext}"
    if args.format == "parquet":
        writer = _ParquetWriter(
            out_path,
            n_codebooks=n_cb,
            n_token_frames=n_token_frames,
            chunk_frames=args.chunk_frames,
            target_fps=cfg.target_fps,
            checkpoint=source_name,
            row_group_size=args.row_group_size,
            zstd_level=args.zstd_level,
        )
    else:
        writer = _CsvWriter(out_path)

    # ── Tokenize loop ──────────────────────────────────────────────────────
    # Each clip is encoded in one full-sequence forward pass. We then split the
    # resulting token timeline into fixed-size rows and buffer those rows for
    # disk writes.
    buf_codes: list[np.ndarray] = []
    buf_meta: list[tuple[str, str]] = []
    n_chunks_written = 0
    n_clips_ok = 0
    n_clips_skipped = 0
    skip_reasons: dict[str, int] = {}
    wan_root = Path(args.wan_root) if args.wan_root else None

    pbar = tqdm(
        loader, desc=f"rank{rank} files", disable=not accelerator.is_local_main_process,
        total=len(loader), smoothing=0.05,
    )

    def flush(batch_size: int) -> None:
        nonlocal buf_codes, buf_meta, n_chunks_written
        if batch_size <= 0:
            return
        # (B, n_token_frames, K), int16
        codes = np.stack(buf_codes[:batch_size], axis=0).astype(np.int16, copy=False)
        clip_ids = [m[0] for m in buf_meta[:batch_size]]
        chunk_paths = [m[1] for m in buf_meta[:batch_size]]
        writer.write_batch(clip_ids, chunk_paths, codes)
        n_chunks_written += batch_size
        del buf_codes[:batch_size]
        del buf_meta[:batch_size]

    for batch in pbar:
        for item in batch:
            if not item["valid"]:
                n_clips_skipped += 1
                skip_reasons[item.get("reason", "unknown")] = (
                    skip_reasons.get(item.get("reason", "unknown"), 0) + 1
                )
                continue
            segment = item["segment"]
            clip_id = item["clip_id"]

            kp = item["keypoints"].unsqueeze(0).to(device, non_blocking=True)
            with torch.inference_mode():
                codes_list = model.tokenize(kp)
            # (T_codes, K), int16. Split this token sequence into fixed rows.
            codes = (
                torch.stack(codes_list, dim=-1)
                .squeeze(0)
                .to(torch.int16)
                .cpu()
                .numpy()
            )
            n_token_chunks = codes.shape[0] // n_token_frames
            if wan_root is not None:
                existing_wan_chunks = _count_existing_wan_chunks(
                    wan_root, clip_id, segment,
                )
                if existing_wan_chunks == 0:
                    if args.require_wan_chunks:
                        n_clips_skipped += 1
                        skip_reasons["missing_wan_chunks"] = (
                            skip_reasons.get("missing_wan_chunks", 0) + 1
                        )
                        continue
                    # No Wan match exists, but the caller did not require one:
                    # encode all token chunks derived from the pose file.
                else:
                    # Wan chunks are the join target, so never emit more pose
                    # token rows than existing <segment>_chunkXXXX files.
                    n_token_chunks = min(n_token_chunks, existing_wan_chunks)

            if n_token_chunks == 0:
                n_clips_skipped += 1
                skip_reasons["too_few_token_steps"] = (
                    skip_reasons.get("too_few_token_steps", 0) + 1
                )
                continue

            n_clips_ok += 1
            for c in range(n_token_chunks):
                start = c * n_token_frames
                end = start + n_token_frames
                buf_codes.append(np.ascontiguousarray(codes[start:end]))
                buf_meta.append((clip_id, f"{segment}_chunk{c:04d}"))

        while len(buf_codes) >= args.batch_size:
            flush(args.batch_size)

        if n_chunks_written and accelerator.is_local_main_process:
            pbar.set_postfix(
                clips=n_clips_ok, chunks=n_chunks_written,
                skipped=n_clips_skipped, refresh=False,
            )

    # Drain remainder (the last partial batch).
    while buf_codes:
        flush(len(buf_codes))

    writer.close()

    print(
        f"[rank {rank}]  wrote {n_chunks_written:,} chunk rows  "
        f"({n_clips_ok:,} clips ok, {n_clips_skipped:,} skipped)  "
        f"→ {out_path}"
    )
    if skip_reasons:
        print(f"[rank {rank}]  skip reasons: {skip_reasons}")

    # ── Summary on rank 0 ──────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shards = sorted(output_dir.glob(f"rank*.{ext}"))
        total_bytes = sum(p.stat().st_size for p in shards)
        if args.format == "parquet":
            accelerator.print(
                f"\nDone. {len(shards)} parquet shard(s) → {output_dir}  "
                f"({total_bytes / 1e6:.1f} MB total)\n"
                f"Load with:\n"
                f"  import pandas as pd, glob\n"
                f"  df = pd.concat(pd.read_parquet(f) for f in "
                f"glob.glob('{output_dir}/rank*.parquet')).set_index('chunk_path')\n"
                f"  tokens = df.loc[chunk]['tokens'].reshape({n_cb}, {n_token_frames})"
            )
        else:
            accelerator.print(
                f"\nDone. {len(shards)} per-rank CSV(s) → {output_dir}  "
                f"({total_bytes / 1e6:.1f} MB total)\n"
                f"Concat with:\n"
                f"  head -n 1 {output_dir}/rank00.csv > {output_dir}/all.csv\n"
                f"  for f in {output_dir}/rank*.csv; do tail -n +2 $f >> {output_dir}/all.csv; done"
            )


if __name__ == "__main__":
    main()
