"""Re-render comparison videos for a trained checkpoint with the fixed pipeline.

What was wrong before:
  • prepare_test_clips loaded at source_fps and fed wrong-fps content to the model
  • render_test_comparisons fed the full clip (T~600) to a model trained on T=80
  • render_comparison_video read source video frames sequentially, ignoring fps

What this script does:
  1. Load the checkpoint's TrainConfig from <run_dir>/config.yaml
  2. Build test clips via prepare_test_clips with source_fps→target_fps resampling
  3. Run CHUNKED inference (T=segment_frames windows) on each clip
  4. Render with the time-aligned video reader

Outputs go to <checkpoint_dir>/comparisons_fixed/ so the originals are preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pose_tokenizer import PoseTokenizerModel
from pose_tokenizer.config import TrainConfig
from pose_tokenizer.data.rendering import prepare_test_clips, render_test_comparisons


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint_dir", help="e.g. checkpoints/baseline-motion-w2/step_15000")
    p.add_argument("--test-dir", default=None, help="defaults to cfg.test_dir")
    p.add_argument("--out", default=None, help="defaults to <checkpoint_dir>/comparisons_fixed")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-clips", type=int, default=None, help="cap for quick checks")
    p.add_argument("--chunk-stride", type=int, default=0,
                   help="Inference chunk stride. 0 → chunk//2 (Hann blend). Set = chunk for legacy.")
    return p.parse_args()


def main():
    args = parse()
    ckpt_dir = Path(args.checkpoint_dir)
    run_dir = ckpt_dir.parent
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml at {cfg_path}")
    cfg, _ = TrainConfig.from_yaml(cfg_path)

    test_dir = Path(args.test_dir) if args.test_dir else (Path(cfg.test_dir) if cfg.test_dir else Path("test_set"))
    out_dir = Path(args.out) if args.out else (ckpt_dir / "comparisons_fixed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run: {run_dir.name}")
    print(f"  checkpoint: {ckpt_dir}")
    print(f"  test_dir:   {test_dir}")
    print(f"  source_fps: {cfg.source_fps}  target_fps: {cfg.target_fps}  segment_frames: {cfg.segment_frames}")
    print(f"  out:        {out_dir}")

    model = PoseTokenizerModel.from_pretrained(str(ckpt_dir)).to(args.device).eval()
    print(f"  params:     {sum(p.numel() for p in model.parameters()):,}")

    target_fps = cfg.target_fps or cfg.source_fps
    max_render_frames = int(cfg.test_max_seconds * target_fps)
    n_clips = args.n_clips if args.n_clips is not None else cfg.n_test_clips
    test_clips = prepare_test_clips(
        data_dir=cfg.data_dir,
        dims={},                            # not needed — test npz files have orig_h/orig_w baked in
        file_paths=[],
        test_dir=test_dir,
        n_clips=n_clips,
        max_frames=max_render_frames,
        source_fps=cfg.source_fps,
        target_fps=target_fps,
    )
    print(f"  test clips: {len(test_clips)}")

    chunk = cfg.segment_frames if cfg.segment_frames > 0 else None
    stride = args.chunk_stride if args.chunk_stride > 0 else None
    paths = render_test_comparisons(
        model, test_clips, out_dir,
        device=args.device, fps=target_fps, max_seconds=cfg.test_max_seconds,
        chunk_size=chunk, chunk_stride=stride,
    )
    print(f"Wrote {len(paths)} videos → {out_dir}")


if __name__ == "__main__":
    main()
