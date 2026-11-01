#!/usr/bin/env python3
"""Head-to-head L1 evaluation of one or more checkpoints on test_set/.

For each checkpoint, runs a single full-clip forward pass on every
``test_set/<actor>/segment_*_poses.npz`` and aggregates:

  * ``rec_l1``   — uniform per-feature L1 over all frames (matches train obj).
  * ``rec_body`` — confidence-masked L1 over body keypoints
                   (joints with detector confidence ≥ --conf-threshold).
  * ``rec_hand`` — same, hand keypoints.
  * ``vel_recovery_body/hand`` — 1 − |vel_err| / |true_motion| on the same
                                 confident subset. 1.0 = perfect motion,
                                 ≤0 = static collapse.

Usage:
    uv run python scripts/eval_on_test_set.py \\
        --checkpoint checkpoints/wide/step_42500/checkpoint.pt \\
        --checkpoint checkpoints/wide-l2norm/step_25000/checkpoint.pt

    # Per-clip breakdown:
    uv run python scripts/eval_on_test_set.py \\
        --checkpoint checkpoints/wide/step_42500/checkpoint.pt --verbose

Single-GPU intentionally: test_set/ is tiny (~38 segments × 12 actors), and
keeping inference single-process makes the metrics deterministic and avoids
the val-rank-aggregation pitfall this script was written to sanity-check.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pose_tokenizer import PoseTokenizerModel
from pose_tokenizer.config import PoseTokenizerConfig, TrainConfig
from pose_tokenizer.data.dataset import load_clip, _resample_clip
from pose_tokenizer.data.rendering import _reconstruct_clip


_HF_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
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
    """Return (model_with_weights_loaded, train_cfg, display_name)."""
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

    if not Path(spec).is_file():
        raise FileNotFoundError(f"Local checkpoint not found: {spec}")
    cfg_path = Path(spec).resolve().parent.parent / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Could not find training config at {cfg_path}. "
            f"Expected layout: <run_dir>/step_N/checkpoint.pt + <run_dir>/config.yaml."
        )
    cfg, overrides = TrainConfig.from_yaml(cfg_path)
    model = (PoseTokenizerModel(config=PoseTokenizerConfig(**overrides))
             if overrides else PoseTokenizerModel())
    ckpt = torch.load(spec, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model, cfg, spec


def load_dims_csv(path: str | Path | None) -> dict[str, tuple[float, float]]:
    if not path:
        return {}
    import csv as _csv
    out: dict[str, tuple[float, float]] = {}
    with open(path) as f:
        for row in _csv.DictReader(f):
            out[row["clip_stem"]] = (float(row["height"]), float(row["width"]))
    return out


def gather_test_clips(test_dir: Path, dims: dict, source_fps: int, target_fps: int) -> list[dict]:
    """Load every test_set/<actor>/segment_*_poses.npz and resample to model fps."""
    clips: list[dict] = []
    for npz in sorted(test_dir.rglob("*_poses.npz")):
        stem = npz.stem
        oh, ow = dims.get(stem, (None, None))
        clip = load_clip(npz, orig_h=oh, orig_w=ow)
        if clip is None:
            continue
        clip = _resample_clip(clip, source_fps, target_fps)
        rel_parent = npz.parent.relative_to(test_dir)
        base = stem.removesuffix("_poses")
        clip_id = base if rel_parent == Path(".") else f"{'__'.join(rel_parent.parts)}__{base}"
        clips.append({**clip, "clip_id": clip_id})
    return clips


def evaluate_checkpoint(
    spec: str, clips: list[dict], device: torch.device,
    *, conf_threshold: float, kp_dim: int, n_body_joints: int,
) -> dict:
    """Run reconstruction on every clip, return aggregated metrics + per-clip rows."""
    model, _cfg, name = resolve_model_and_config(spec)
    model.to(device).eval()

    body_end = n_body_joints * kp_dim          # last body flat index (exclusive)
    n_codebooks = getattr(model.config, "n_codebooks", 0)
    cb_size = model.config.codebook_size

    # Accumulators: (numerator_sum, denominator_count)
    rec_l1_sum = 0.0
    rec_l1_count = 0
    rec_body = [0.0, 0.0]
    rec_hand = [0.0, 0.0]
    vel_err_body = [0.0, 0.0]
    vel_err_hand = [0.0, 0.0]
    true_mot_body = [0.0, 0.0]
    true_mot_hand = [0.0, 0.0]
    # Per-codebook boolean "seen" mask. Token-level resolution: index_fill on
    # each batch's unique codes is O(unique * 1), much cheaper than collecting
    # then taking torch.unique at the end (and trivially DDP-reducible — same
    # representation as the fixed validate()).
    seen_codes = [torch.zeros(cb_size, dtype=torch.bool, device=device)
                  for _ in range(n_codebooks)]

    per_clip_rows = []
    for clip in clips:
        kp = clip["keypoints"]                                  # (T, F)

        # Also tokenize for cb_usage (separate from the reconstruction pass —
        # _reconstruct_clip may chunk, which would inflate the token count and
        # make code-coverage estimates dependent on chunk overlap. For cb_usage
        # we want one set of tokens per frame, so we do a single forward.)
        if n_codebooks > 0:
            kp_t = torch.from_numpy(kp).float().unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(kp_t)
            codes = out.get("codes")
            if codes is not None:
                for cb_idx, codes_i in enumerate(codes):
                    seen_codes[cb_idx].index_fill_(0, codes_i.flatten().unique(), True)

        recon = _reconstruct_clip(model, kp, device=device)     # (T, F) np
        diff = np.abs(recon - kp)
        vel_recon = recon[1:] - recon[:-1]
        vel_true = kp[1:] - kp[:-1]
        vel_diff = np.abs(vel_recon - vel_true)
        true_mag = np.abs(vel_true)

        # ── per-clip uniform rec_l1 ──
        clip_rec_l1 = diff.mean()
        rec_l1_sum += float(diff.sum())
        rec_l1_count += diff.size

        # ── confidence mask ──
        conf = clip.get("confidence")
        if conf is not None and conf_threshold > 0.0:
            m_joint = (conf >= conf_threshold).astype(np.float32)
            m_xy = np.repeat(m_joint, kp_dim, axis=-1)
        else:
            m_xy = np.ones_like(kp)
        m_body, m_hand = m_xy[:, :body_end], m_xy[:, body_end:]

        rec_body[0] += float((diff[:, :body_end] * m_body).sum())
        rec_body[1] += float(m_body.sum())
        rec_hand[0] += float((diff[:, body_end:] * m_hand).sum())
        rec_hand[1] += float(m_hand.sum())

        # Velocity entries: both endpoint frames confident.
        vm_xy = m_xy[1:] * m_xy[:-1]
        vm_body, vm_hand = vm_xy[:, :body_end], vm_xy[:, body_end:]
        vel_err_body[0] += float((vel_diff[:, :body_end] * vm_body).sum())
        vel_err_body[1] += float(vm_body.sum())
        vel_err_hand[0] += float((vel_diff[:, body_end:] * vm_hand).sum())
        vel_err_hand[1] += float(vm_hand.sum())
        true_mot_body[0] += float((true_mag[:, :body_end] * vm_body).sum())
        true_mot_body[1] += float(vm_body.sum())
        true_mot_hand[0] += float((true_mag[:, body_end:] * vm_hand).sum())
        true_mot_hand[1] += float(vm_hand.sum())

        per_clip_rows.append({
            "clip": clip["clip_id"],
            "frames": kp.shape[0],
            "rec_l1": float(clip_rec_l1),
        })

    def r(p): return p[0] / p[1] if p[1] > 0 else 0.0
    def vrec(err_p, mot_p): return 1.0 - r(err_p) / max(r(mot_p), 1e-9)

    per_cb_usage = [s.sum().item() / cb_size * 100.0 for s in seen_codes]
    return {
        "name": name,
        "rec_l1": rec_l1_sum / max(rec_l1_count, 1),
        "rec_body": r(rec_body),
        "rec_hand": r(rec_hand),
        "vel_recovery_body": vrec(vel_err_body, true_mot_body),
        "vel_recovery_hand": vrec(vel_err_hand, true_mot_hand),
        "cb_usage": (sum(per_cb_usage) / len(per_cb_usage)) if per_cb_usage else 0.0,
        "cb_usage_per": per_cb_usage,
        "per_clip": per_clip_rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", action="append", required=True,
                   help="Local checkpoint.pt path OR HF Hub repo ID. Repeatable.")
    p.add_argument("--test-dir", default="test_set",
                   help="Directory with <actor>/<segment>_poses.npz files.")
    p.add_argument("--dims-csv", default="dims.csv",
                   help="CSV with clip_stem,height,width — used to rescale pixel "
                        "coords to shoulder-width-normalised offsets. Defaults "
                        "to <repo>/dims.csv.")
    p.add_argument("--source-fps", type=int, default=30,
                   help="FPS of the raw .npz pose data.")
    p.add_argument("--target-fps", type=int, default=25,
                   help="FPS the model was trained at — clips are resampled "
                        "before reconstruction.")
    p.add_argument("--conf-threshold", type=float, default=0.5,
                   help="Detector confidence ≥ this counts toward masked metrics.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-clip rec_l1 in addition to the summary.")
    return p.parse_args()


def main():
    args = parse_args()

    # Joint layout is fixed by the project (55 joints, 5 face + 8 body + 21 + 21 hands).
    # Take it from the first model's config so the script self-validates.
    first_model, _, _ = resolve_model_and_config(args.checkpoint[0])
    fc = first_model.config
    kp_dim = fc.keypoint_dim
    n_body_joints = (fc.input_features - 42 * kp_dim) // kp_dim
    del first_model

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"--test-dir not found: {test_dir}")

    dims_path = Path(args.dims_csv) if args.dims_csv else None
    if dims_path and not dims_path.is_file():
        print(f"  [warn] dims_csv not found at {dims_path}; falling back to "
              f"per-clip H/W from the .npz files.")
        dims_path = None
    dims = load_dims_csv(dims_path)

    clips = gather_test_clips(test_dir, dims, args.source_fps, args.target_fps)
    if not clips:
        raise RuntimeError(f"No *_poses.npz files found under {test_dir}.")
    total_frames = sum(c["keypoints"].shape[0] for c in clips)
    print(f"test_set: {len(clips)} clips, {total_frames} frames total @ {args.target_fps} fps")
    print(f"          conf_threshold={args.conf_threshold}  "
          f"layout={n_body_joints} body + 42 hand joints (kp_dim={kp_dim})")

    device = torch.device(args.device)
    results = []
    for spec in args.checkpoint:
        print(f"\n→ evaluating {spec}")
        r = evaluate_checkpoint(
            spec, clips, device,
            conf_threshold=args.conf_threshold,
            kp_dim=kp_dim, n_body_joints=n_body_joints,
        )
        results.append(r)
        if args.verbose:
            print(f"  per-clip rec_l1:")
            for row in sorted(r["per_clip"], key=lambda x: x["rec_l1"]):
                print(f"    {row['clip']:<40s}  T={row['frames']:>4d}  "
                      f"rec_l1={row['rec_l1']:.5f}")

    # ── Summary table ─────────────────────────────────────────────────────
    def _disp(name: str) -> str:
        spec = Path(name)
        if not spec.is_file():
            return name
        run = spec.resolve().parent.parent.name
        step = spec.resolve().parent.name
        return f"{run}/{step}"

    name_w = max(len(_disp(r["name"])) for r in results)
    name_w = max(name_w, 6)
    print("\n" + "=" * (name_w + 92))
    print(f"  {'run':<{name_w}s}  {'rec_l1':>8s}  {'rec_body':>9s}  {'rec_hand':>9s}  "
          f"{'vel_rec_body':>13s}  {'vel_rec_hand':>13s}  {'cb_usage':>9s}")
    print("  " + "-" * (name_w + 90))
    for r in results:
        print(f"  {_disp(r['name']):<{name_w}s}  "
              f"{r['rec_l1']:8.5f}  "
              f"{r['rec_body']:9.5f}  "
              f"{r['rec_hand']:9.5f}  "
              f"{r['vel_recovery_body']*100:12.1f}%  "
              f"{r['vel_recovery_hand']*100:12.1f}%  "
              f"{r['cb_usage']:8.1f}%")

    # Per-codebook breakdown only if any run uses VQ.
    if any(r["cb_usage_per"] for r in results):
        print("\n  per-codebook usage (%):")
        n_cb = max(len(r["cb_usage_per"]) for r in results)
        header = "    " + " ".join(f"cb{i:>2d}" for i in range(n_cb))
        print(header)
        for r in results:
            row = "    " + " ".join(
                f"{u:4.0f}" if i < len(r["cb_usage_per"]) else "  --"
                for i, u in enumerate(r["cb_usage_per"] + [0]*(n_cb - len(r["cb_usage_per"])))
            )
            print(f"{row}  ← {_disp(r['name'])}")

    # Highlight the best on each metric.
    if len(results) > 1:
        print()
        for key, lower_is_better in [
            ("rec_l1", True), ("rec_body", True), ("rec_hand", True),
            ("vel_recovery_body", False), ("vel_recovery_hand", False),
            ("cb_usage", False),
        ]:
            ranked = sorted(results, key=lambda r: r[key], reverse=not lower_is_better)
            arrow = "↓" if lower_is_better else "↑"
            print(f"  best {key:<18s} ({arrow}): {_disp(ranked[0]['name'])}")


if __name__ == "__main__":
    main()
