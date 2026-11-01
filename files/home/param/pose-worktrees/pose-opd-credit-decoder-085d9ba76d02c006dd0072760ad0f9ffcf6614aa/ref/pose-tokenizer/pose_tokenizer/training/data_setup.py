"""Build the train/val DataLoaders and the test-clip set for comparison rendering."""

from __future__ import annotations

import random
from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader

from pose_tokenizer.data import PoseDataset, make_collate_fn
from pose_tokenizer.data.dataset import set_first_frame_root
from pose_tokenizer.data.rendering import prepare_test_clips


def build_dataloaders(
    cfg,
    downsampling_factor: int,
    accelerator,
    *,
    include_confidence: bool = False,
):
    """Build train/val DataLoaders and the list of test clips for comparison renders.

    Pipeline:
      1. Build a lightweight ``base_ds`` (no length scan) just to enumerate files.
      2. Deterministically shuffle and split into train/val by file index.
      3. Build the per-split PoseDatasets (each runs the segment-index build,
         hitting the shared length cache).
      4. Wrap each in a DataLoader with the standard worker config.
      5. Prepare test clips from val files (+ optional ``cfg.test_dir`` extras).

    Args:
        cfg: TrainConfig
        downsampling_factor: model.downsampling_factor — needed by collate to
            pad T to a multiple compatible with the encoder.
        accelerator: only used to gate printed info to the main process.

    Returns:
        (train_loader, val_loader, test_clips)
    """
    # Apply the root convention BEFORE any dataset/test-clip construction so
    # training, validation and comparison rendering all share it (DataLoader
    # workers fork after this point and inherit the setting).
    first_frame_root = bool(getattr(cfg, "first_frame_root", False))
    set_first_frame_root(first_frame_root)
    accelerator.print(
        f"Root convention: {'FIXED frame-0' if first_frame_root else 'per-frame'} "
        f"mid-shoulder root (first_frame_root={first_frame_root})"
    )

    seg = cfg.segment_frames if cfg.segment_frames > 0 else None
    mf = cfg.max_frames if (not seg and cfg.max_frames > 0) else None
    rc_train = bool(mf and cfg.random_crop_frames)

    # Resolve dataset spec(s). Multi-dataset (cfg.datasets) OVERRIDES the single
    # data_dir/manifest/dims_csv; each is built as its own self-contained
    # PoseDataset (own root/manifest/dims/source_fps) and composed via
    # ConcatDataset, so per-dataset fps/dims stay correct with no cross-dataset
    # clip-stem collisions.
    specs = list(getattr(cfg, "datasets", None) or [])
    if not specs:
        specs = [{"data_dir": cfg.data_dir, "manifest": cfg.manifest,
                  "dims_csv": cfg.dims_csv, "source_fps": cfg.source_fps}]
    n_specs = len(specs)
    # Split val_count / num_files evenly across datasets (single dataset => legacy).
    per_val = cfg.val_count if n_specs == 1 else max(1, cfg.val_count // n_specs)
    per_numfiles = cfg.num_files if n_specs == 1 else (
        max(1, cfg.num_files // n_specs) if cfg.num_files > 0 else 0
    )

    def _build_one(spec: dict):
        data_dir = spec["data_dir"]
        manifest = Path(spec["manifest"]) if spec.get("manifest") else None
        dims_csv = Path(spec["dims_csv"]) if spec.get("dims_csv") else None
        source_fps = int(spec.get("source_fps") or cfg.source_fps)

        base_ds = PoseDataset(
            data_dir=data_dir, manifest=manifest, dims_csv=dims_csv,
            include_confidence=include_confidence,
        )
        sample = base_ds.files[0]
        if not sample.is_file():
            raise FileNotFoundError(
                f"First manifest entry is not a readable file:\n  {sample}\n"
                f"data_dir={data_dir!r} — paths are data_dir + each manifest line."
            )

        idx = list(range(len(base_ds)))
        random.Random(42).shuffle(idx)
        if per_numfiles > 0:
            idx = idx[:per_numfiles]
        n_val = min(per_val, len(idx) // 2)
        train_idx = idx[n_val:]
        val_idx = idx[:n_val]

        ds_kw = dict(
            data_dir=data_dir, manifest=manifest, dims_csv=dims_csv,
            source_fps=source_fps, target_fps=cfg.target_fps,
            segment_frames=seg, max_frames=mf,
            prefix_frames=getattr(cfg, "prefix_frames", 0),
            include_confidence=include_confidence,
        )
        train_ds = PoseDataset(
            **ds_kw, random_crop=rc_train,
            random_segment_offset=cfg.random_segment_offset, file_indices=train_idx,
        )
        val_ds = PoseDataset(
            **ds_kw, random_crop=False, random_segment_offset=False, file_indices=val_idx,
        )
        return base_ds, train_ds, val_ds, val_idx

    bases, train_parts, val_parts, val_idxs = [], [], [], []
    for spec in specs:
        base_ds, tr, va, vidx = _build_one(spec)
        bases.append(base_ds); train_parts.append(tr); val_parts.append(va); val_idxs.append(vidx)

    if n_specs == 1:
        train_ds, val_ds = train_parts[0], val_parts[0]
    else:
        train_ds, val_ds = ConcatDataset(train_parts), ConcatDataset(val_parts)
        accelerator.print(f"Pooling {n_specs} datasets via ConcatDataset.")

    n_train_files = sum(len(t.files) for t in train_parts)
    n_val_files = sum(len(v.files) for v in val_parts)
    accelerator.print(
        f"Files: {n_train_files:,} train / {n_val_files:,} val  →  "
        f"Samples: {len(train_ds):,} train / {len(val_ds):,} val"
    )
    if seg:
        fps = cfg.target_fps or cfg.source_fps
        pf = getattr(cfg, "prefix_frames", 0)
        if pf > 0:
            read = seg + pf
            # Reads must be a multiple of the downsampling factor; otherwise the
            # collate fn pads every (uniform-length) sample up to a multiple,
            # re-introducing the all-zero frames that NaN the quantizer's L2
            # normalisation in the backward pass.
            if read % downsampling_factor != 0:
                raise ValueError(
                    f"segment_frames + prefix_frames = {read} must be a multiple of the "
                    f"model downsampling_factor ({downsampling_factor}) to avoid zero-padding. "
                    f"Adjust segment_frames/prefix_frames."
                )
            accelerator.print(
                f"segment_frames={seg} supervised ({seg / fps:.1f}s @{fps}fps) "
                f"+ prefix_frames={pf} warmup ({pf / fps:.1f}s) "
                f"= {read} read ({read / fps:.1f}s), fixed-length (no padding)"
            )
        else:
            accelerator.print(f"segment_frames={seg} ({seg / fps:.1f}s @{fps}fps)")
    elif mf:
        crop_msg = "random temporal window" if rc_train else "prefix [:max_frames]"
        accelerator.print(f"max_frames={mf} (train: {crop_msg}; val: prefix)")

    collate_fn = make_collate_fn(downsampling_factor)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=6 if cfg.num_workers > 0 else None,
    )
    # Validation: separate, NON-persistent workers. Val only runs every
    # `validation_interval` steps over ~100 files; keeping 10 workers alive
    # forever (as in the original shared loader_kw) leaks ~10 idle processes
    # that compete with train workers for NFS bandwidth + RAM. We rebuild on
    # each validate() call — re-fork is <1s vs. >>multi-second savings during
    # training.
    val_bs = cfg.val_batch_size if cfg.val_batch_size > 0 else cfg.batch_size
    val_workers = min(4, cfg.num_workers)
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=False,
        prefetch_factor=2 if val_workers > 0 else None,
    )

    # Test clips for the comparison videos rendered at every save step.
    # Resampled to target_fps so the model sees the same time base as training;
    # max_render_frames is therefore in target_fps units.
    max_render_frames = int(cfg.test_max_seconds * (cfg.target_fps or cfg.source_fps))
    # Gather val paths + merged dims across all datasets for the comparison renders.
    val_paths = [base.files[i] for base, vidx in zip(bases, val_idxs) for i in vidx]
    merged_dims: dict = {}
    for base in bases:
        merged_dims.update(base.dims)
    test_clips = prepare_test_clips(
        data_dir=specs[0]["data_dir"],
        dims=merged_dims,
        file_paths=val_paths,
        test_dir=cfg.test_dir or None,
        n_clips=cfg.n_test_clips,
        max_frames=max_render_frames,
        source_fps=cfg.source_fps,
        target_fps=cfg.target_fps or cfg.source_fps,
    )
    accelerator.print(f"Test clips for rendering: {len(test_clips)}")

    return train_loader, val_loader, test_clips
