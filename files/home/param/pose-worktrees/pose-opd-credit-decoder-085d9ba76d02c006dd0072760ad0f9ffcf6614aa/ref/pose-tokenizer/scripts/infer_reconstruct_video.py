#!/usr/bin/env python3
"""End-to-end pose-tokenizer inference on an MP4 or RTMW-X pose NPZ.

Examples:
    uv run python scripts/infer_reconstruct_video.py \
        checkpoints/wide-l2norm-confidence-0.25-wn/step_140000 IMG_1781_30fps.mp4

    uv run python scripts/infer_reconstruct_video.py \
        InternalCan/base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048 \
        pose_overlays/IMG_1781_30fps_poses.npz --video IMG_1781_30fps.mp4
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import re
import site
import sys
from pathlib import Path


def _preload_cuda12_libs() -> list[str]:
    """Preload pip-installed CUDA 12 libs so ONNX Runtime can find them."""
    loaded: list[str] = []
    sp_dirs = site.getsitepackages() + [site.getusersitepackages()]
    seen: set[str] = set()
    for sp_dir in sp_dirs:
        for lib in sorted(glob.glob(os.path.join(sp_dir, "nvidia/*/lib/*.so*"))):
            base = os.path.basename(lib)
            if base in seen:
                continue
            seen.add(base)
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                loaded.append(lib)
            except OSError:
                pass
    return loaded


_PRELOADED_CUDA_LIBS = _preload_cuda12_libs()

import cv2
import numpy as np
import torch
from rtmlib import PoseTracker, Wholebody

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pose_tokenizer import PoseTokenizerModel  # noqa: E402
from pose_tokenizer.config import PoseTokenizerConfig, TrainConfig  # noqa: E402
from pose_tokenizer.data.kinematic import (  # noqa: E402
    NUM_JOINTS,
    ROOT_LEFT,
    ROOT_RIGHT,
    SELECTED_INDICES,
    positions_to_offsets,
)
from pose_tokenizer.data.rendering import (  # noqa: E402
    _reconstruct_clip,
    render_comparison_video,
    render_overlay_video,
)
from rtmwx_overlay import OneEuroKeypointSmoother  # noqa: E402

_CKPT_EXTS = (".pt", ".pth", ".bin", ".safetensors")
_HF_REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract RTMW-X poses, reconstruct with a pose tokenizer, and render a side-by-side MP4.",
    )
    p.add_argument("checkpoint", help="HF repo, checkpoint dir, checkpoint.pt, or run dir containing config.yaml + step_*")
    p.add_argument("input", type=Path, help="Input .mp4 or RTMW-X pose .npz")
    p.add_argument("--video", type=Path, default=None, help="Source video for .npz inputs")
    p.add_argument("--out", type=Path, default=None, help="Output comparison MP4 (Original | Reconstructed)")
    p.add_argument("--overlay-out", type=Path, default=None,
                   help="Output overlay MP4 (both skeletons on one frame); defaults to <out>_overlay.mp4")
    p.add_argument("--pose-out", type=Path, default=None, help="Where to save extracted RTMW-X poses for .mp4 inputs")
    p.add_argument("--overwrite-poses", action="store_true", help="Re-extract RTMW-X poses even if the pose NPZ exists")
    p.add_argument(
        "--npz-coord-order",
        default="auto",
        choices=["auto", "xy", "yx"],
        help="Coordinate order in input NPZ keypoints/poses. RTMW keypoints are xy; some saved poses are yx.",
    )
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rtmw-device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rtmw-backend", default="onnxruntime")
    p.add_argument("--rtmw-mode", default="balanced", choices=["performance", "balanced", "lightweight"])
    p.add_argument("--det-frequency", type=int, default=1)
    p.add_argument("--euro-min-cutoff", type=float, default=3.0)
    p.add_argument("--euro-beta", type=float, default=0.5)
    p.add_argument("--euro-d-cutoff", type=float, default=1.0)
    p.add_argument("--conf-thresh", type=float, default=0.5)
    p.add_argument("--conf-lo", type=float, default=1.0,
                   help="RTMW score mapped to green (unconfident) in the overlay gradient.")
    p.add_argument("--conf-hi", type=float, default=5.0,
                   help="RTMW score mapped to red (confident) in the overlay gradient.")
    p.add_argument("--deshift", type=int, default=-1,
                   help="Advance recon by N frames to undo the causal_loss_shift latency "
                        "(recon[t] reconstructs gt[t-N]). -1 = auto (use cfg.causal_loss_shift). "
                        "0 = disable.")
    p.add_argument("--pad-mode", default="constant", choices=["constant", "replicate", "reflect"],
                   help="Left-padding mode for causal convs. 'constant' = zero-pad (default/as "
                        "trained; optimal). 'replicate'/'reflect' are diagnostics only: on the "
                        "correctly-trained (pad_left=1) code they hurt the opening frames.")
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--chunk-stride", type=int, default=0, help="0 means config inference_chunk_stride or chunk_size//2")
    return p.parse_args()


def is_hf_repo_id(spec: str) -> bool:
    if Path(spec).exists() or spec.startswith(("/", "./", "../")) or spec.endswith(_CKPT_EXTS):
        return False
    return bool(_HF_REPO_PATTERN.match(spec))


def _latest_step_dir(run_dir: Path) -> Path:
    def step_num(path: Path) -> int:
        try:
            return int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    candidates = [
        p for p in run_dir.glob("step_*")
        if p.is_dir() and ((p / "config.json").is_file() or (p / "model.safetensors").is_file())
    ]
    if not candidates:
        raise FileNotFoundError(f"No HF-format step_* checkpoint found under {run_dir}")
    return max(candidates, key=step_num)


def resolve_model_and_config(spec: str, device: str) -> tuple[PoseTokenizerModel, TrainConfig, str, Path | None]:
    """Load model via from_pretrained and the adjacent training config.yaml."""
    if is_hf_repo_id(spec):
        from huggingface_hub import hf_hub_download

        model = PoseTokenizerModel.from_pretrained(spec).to(device).eval()
        cfg_path = hf_hub_download(repo_id=spec, filename="config.yaml")
        cfg, _ = TrainConfig.from_yaml(cfg_path)
        return model, cfg, spec, None

    path = Path(spec).expanduser().resolve()
    ckpt_dir = path.parent if path.is_file() else path
    if (ckpt_dir / "config.yaml").is_file() and not (ckpt_dir / "config.json").is_file():
        ckpt_dir = _latest_step_dir(ckpt_dir)

    cfg_path = ckpt_dir / "config.yaml"
    if not cfg_path.is_file():
        cfg_path = ckpt_dir.parent / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Could not find config.yaml for {spec}. Expected it in the checkpoint dir or its parent run dir.",
        )

    cfg, model_overrides = TrainConfig.from_yaml(cfg_path)
    if (ckpt_dir / "config.json").is_file() or (ckpt_dir / "model.safetensors").is_file():
        model = PoseTokenizerModel.from_pretrained(str(ckpt_dir)).to(device).eval()
    else:
        ckpt_file = path if path.is_file() else ckpt_dir / "checkpoint.pt"
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"Could not find HF files or checkpoint.pt at {ckpt_dir}")
        model = (
            PoseTokenizerModel(config=PoseTokenizerConfig(**model_overrides))
            if model_overrides else PoseTokenizerModel()
        )
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()

    return model, cfg, str(ckpt_dir), ckpt_dir


def load_norm_stats(ckpt_dir: Path | None, input_features: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if ckpt_dir is None:
        return None, None
    ns_path = ckpt_dir / "norm_stats.npz"
    if not ns_path.is_file():
        return None, None
    stats = np.load(ns_path)
    center = stats["center"].astype(np.float32)
    scale = stats["scale"].astype(np.float32)
    if center.shape[-1] != input_features or scale.shape[-1] != input_features:
        raise ValueError(
            f"{ns_path} has {center.shape[-1]} features, but model expects {input_features}.",
        )
    return center, scale


def extract_rtmw_poses(video_path: Path, out_path: Path, args: argparse.Namespace) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"Extracting RTMW-X: {frame_count} frames @ {fps:.2f} fps ({width}x{height}) "
        f"device={args.rtmw_device}",
        flush=True,
    )
    if args.rtmw_backend == "onnxruntime" and str(args.rtmw_device).startswith("cuda"):
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "RTMW-X was requested on CUDA, but onnxruntime does not expose "
                f"CUDAExecutionProvider. Available providers: {providers}. "
                "Remove the CPU wheel with `uv pip uninstall onnxruntime`, leaving "
                "`onnxruntime-gpu` installed, then rerun.",
            )
    tracker = PoseTracker(
        Wholebody,
        det_frequency=args.det_frequency,
        to_openpose=False,
        mode=args.rtmw_mode,
        backend=args.rtmw_backend,
        device=args.rtmw_device,
    )
    smoother = OneEuroKeypointSmoother(
        fps=fps,
        min_cutoff=args.euro_min_cutoff,
        beta=args.euro_beta,
        d_cutoff=args.euro_d_cutoff,
    )

    all_keypoints: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    last_kp: np.ndarray | None = None
    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, scores = tracker(frame)
        if keypoints is not None and len(keypoints) > 0:
            best = 0 if len(keypoints) == 1 else int(np.argmax([np.asarray(s).mean() for s in scores]))
            kp, sc = np.asarray(keypoints[best]), np.asarray(scores[best])
            kp, sc = smoother(kp, sc, conf_thresh=args.conf_thresh)
            last_kp = kp.copy()
        elif last_kp is not None:
            kp = last_kp.copy()
            sc = np.zeros(133, dtype=np.float32)
        else:
            kp = np.zeros((133, 2), dtype=np.float32)
            sc = np.zeros(133, dtype=np.float32)

        all_keypoints.append(kp.astype(np.float32, copy=False))
        all_scores.append(sc.astype(np.float32, copy=False))
        fi += 1
        if fi % 50 == 0:
            print(f"  {fi}/{frame_count} frames", flush=True)

    cap.release()
    keypoints_arr = np.stack(all_keypoints)
    scores_arr = np.stack(all_scores)
    poses = np.concatenate([keypoints_arr, scores_arr[..., None]], axis=-1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        keypoints=keypoints_arr,
        scores=scores_arr,
        poses=poses,
        fps=fps,
        resolution=np.array([width, height], dtype=np.int32),
        orig_w=np.array(width, dtype=np.float32),
        orig_h=np.array(height, dtype=np.float32),
    )
    print(f"Saved poses -> {out_path}", flush=True)
    return out_path


def load_raw_pose_npz(path: Path) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    with np.load(path) as data:
        if "keypoints" in data and "scores" in data:
            keypoints = np.asarray(data["keypoints"], dtype=np.float32)
            scores = np.asarray(data["scores"], dtype=np.float32)
        elif "poses" in data:
            poses = np.asarray(data["poses"], dtype=np.float32)
            keypoints, scores = poses[..., :2], poses[..., 2]
        else:
            raise ValueError(f"{path} must contain keypoints/scores or poses.")

        fps = float(data["fps"]) if "fps" in data else 30.0
        if "resolution" in data:
            width, height = [int(x) for x in data["resolution"]]
        else:
            width = int(data["orig_w"])
            height = int(data["orig_h"])

    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 2):
        raise ValueError(f"Expected keypoints shape (T, 133, 2), got {keypoints.shape}")
    if scores.shape != keypoints.shape[:2]:
        raise ValueError(f"Expected scores shape {keypoints.shape[:2]}, got {scores.shape}")
    return keypoints, scores, fps, width, height


def preprocess_to_features(
    raw_kp: np.ndarray,
    raw_sc: np.ndarray,
    height: int,
    width: int,
    coord_order: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_kp = raw_kp[:, SELECTED_INDICES]
    selected_sc = raw_sc[:, SELECTED_INDICES].astype(np.float32)

    positions = np.empty_like(selected_kp, dtype=np.float32)
    if coord_order == "xy":
        positions[..., 0] = selected_kp[..., 1] / float(height)
        positions[..., 1] = selected_kp[..., 0] / float(width)
    elif coord_order == "yx":
        positions[..., 0] = selected_kp[..., 0] / float(height)
        positions[..., 1] = selected_kp[..., 1] / float(width)
    else:
        raise ValueError(f"coord_order must be 'xy' or 'yx', got {coord_order!r}")

    shoulder_width = np.abs(positions[:, ROOT_LEFT, 1] - positions[:, ROOT_RIGHT, 1])
    median_sw = float(np.nanmedian(shoulder_width))
    if not np.isfinite(median_sw) or median_sw <= 0:
        raise ValueError("Invalid shoulder width from extracted poses; check RTMW-X detections.")

    root = (positions[:, ROOT_LEFT, :] + positions[:, ROOT_RIGHT, :]) / 2.0
    _, offsets = positions_to_offsets(positions)
    keypoints_flat = (offsets / median_sw).reshape(positions.shape[0], -1).astype(np.float32)
    global_state = np.stack(
        [root[:, 0], root[:, 1], np.full(positions.shape[0], median_sw, dtype=np.float32)],
        axis=-1,
    ).astype(np.float32)
    return keypoints_flat, selected_sc, global_state


def infer_coord_order(raw_kp: np.ndarray, height: int, width: int, requested: str) -> str:
    if requested != "auto":
        return requested

    sample = raw_kp[..., :2].reshape(-1, 2)
    sample = sample[np.isfinite(sample).all(axis=1)]
    if sample.size == 0:
        return "xy"

    margin = 0.05
    xy_valid = (
        (sample[:, 0] >= -margin * width)
        & (sample[:, 0] <= (1.0 + margin) * width)
        & (sample[:, 1] >= -margin * height)
        & (sample[:, 1] <= (1.0 + margin) * height)
    ).mean()
    yx_valid = (
        (sample[:, 0] >= -margin * height)
        & (sample[:, 0] <= (1.0 + margin) * height)
        & (sample[:, 1] >= -margin * width)
        & (sample[:, 1] <= (1.0 + margin) * width)
    ).mean()

    return "yx" if yx_valid > xy_valid else "xy"


def resample_to_fps(arrays: list[np.ndarray], src_fps: float, target_fps: float) -> list[np.ndarray]:
    if abs(src_fps - target_fps) <= 0.01:
        return arrays
    source_t = arrays[0].shape[0]
    target_t = max(1, round(source_t * target_fps / src_fps))
    idx = np.round(np.linspace(0, source_t - 1, target_t)).astype(int)
    print(f"Resampled {src_fps:.2f} -> {target_fps:.2f} fps: T {source_t} -> {target_t}", flush=True)
    return [arr[idx] for arr in arrays]


def make_model_input(keypoints_flat: np.ndarray, confidence: np.ndarray, input_features: int) -> np.ndarray:
    if input_features == NUM_JOINTS * 2:
        return keypoints_flat
    if input_features == NUM_JOINTS * 3:
        coords = keypoints_flat.reshape(-1, NUM_JOINTS, 2)
        return np.concatenate([coords, confidence[..., None]], axis=-1).reshape(-1, NUM_JOINTS * 3).astype(np.float32)
    raise ValueError(
        f"Unsupported model input_features={input_features}; expected {NUM_JOINTS * 2} or {NUM_JOINTS * 3}.",
    )


def split_model_output(reconstruction: np.ndarray, input_features: int) -> tuple[np.ndarray, np.ndarray | None]:
    if input_features == NUM_JOINTS * 2:
        return reconstruction.astype(np.float32), None
    recon = reconstruction.reshape(-1, NUM_JOINTS, 3)
    coords = recon[..., :2].reshape(reconstruction.shape[0], NUM_JOINTS * 2).astype(np.float32)
    conf = np.clip(recon[..., 2], 0.0, 1.0).astype(np.float32)
    return coords, conf


def run_forward(
    model: PoseTokenizerModel,
    model_input: np.ndarray,
    device: str,
    *,
    chunk_size: int | None,
    stride: int | None,
    norm_center: np.ndarray | None,
    norm_scale: np.ndarray | None,
) -> np.ndarray:
    x = model_input
    if norm_center is not None and norm_scale is not None:
        x = (x - norm_center) / norm_scale
    with torch.no_grad():
        recon = _reconstruct_clip(model, x.astype(np.float32, copy=False), device, chunk_size=chunk_size, stride=stride)
    if norm_center is not None and norm_scale is not None:
        recon = recon * norm_scale + norm_center
    return recon.astype(np.float32, copy=False)


def save_encoded_tokens(
    model: PoseTokenizerModel,
    model_input: np.ndarray,
    device: str,
    out_path: Path,
    *,
    norm_center: np.ndarray | None,
    norm_scale: np.ndarray | None,
    checkpoint: str,
    pose_path: Path,
    target_fps: float,
) -> Path:
    x = model_input
    if norm_center is not None and norm_scale is not None:
        x = (x - norm_center) / norm_scale

    with torch.no_grad():
        tokens = model.tokenize(torch.from_numpy(x.astype(np.float32, copy=False)).unsqueeze(0).to(device))

    token_path = out_path.with_suffix(".pt")
    token_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "codes": [code.squeeze(0).cpu() for code in tokens] if tokens is not None else None,
            "checkpoint": checkpoint,
            "pose_path": str(pose_path),
            "target_fps": target_fps,
            "input_features": model.config.input_features,
            "n_codebooks": model.config.n_codebooks,
            "codebook_size": model.config.codebook_size,
            "num_frames": int(model_input.shape[0]),
            "encoding_mode": "full_clip_single_pass",
        },
        token_path,
    )
    return token_path


def default_output_path(input_path: Path, model_name: str) -> Path:
    safe_model = Path(model_name).name.replace("/", "-")
    return Path("pose_comparisons") / f"{input_path.stem}__{safe_model}_comparison.mp4"


def main() -> None:
    args = parse_args()
    if args.pad_mode != "constant":
        # Swap the causal convs' left-padding mode (mitigates the zero-pad cold-start).
        import torch.nn.functional as _F
        from pose_tokenizer.model import layers as _layers
        _mode = args.pad_mode

        def _causal_forward(self, x):
            return self.conv(_F.pad(x, (self.pad_left, 0), mode=_mode))

        _layers.CausalConv1d.forward = _causal_forward
        print(f"[pad] causal convs using left pad mode = {_mode}", flush=True)
    model, cfg, model_name, ckpt_dir = resolve_model_and_config(args.checkpoint, args.device)
    input_features = model.config.input_features
    target_fps = float(cfg.target_fps or cfg.source_fps or 25)
    chunk_size = cfg.segment_frames if cfg.segment_frames > 0 else None
    stride = args.chunk_stride or cfg.inference_chunk_stride or None

    print(
        f"Loaded {model_name}: input_features={input_features}, target_fps={target_fps:g}, "
        f"chunk_size={chunk_size}, stride={stride or 'default'}",
        flush=True,
    )

    input_path = args.input
    video_path = args.video
    if input_path.suffix.lower() == ".mp4":
        video_path = input_path
        pose_path = args.pose_out or Path("pose_overlays") / f"{input_path.stem}_rtmwx_poses.npz"
        if pose_path.exists() and not args.overwrite_poses:
            print(f"Using existing poses -> {pose_path}", flush=True)
        else:
            extract_rtmw_poses(input_path, pose_path, args)
    else:
        pose_path = input_path

    raw_kp, raw_sc, src_fps, width, height = load_raw_pose_npz(pose_path)
    coord_order = infer_coord_order(raw_kp, height, width, args.npz_coord_order)
    print(f"Using NPZ coordinate order: {coord_order}", flush=True)
    keypoints_flat, confidence, global_state = preprocess_to_features(raw_kp, raw_sc, height, width, coord_order)
    keypoints_flat, confidence, global_state = resample_to_fps(
        [keypoints_flat, confidence, global_state],
        src_fps,
        target_fps,
    )

    model_input = make_model_input(keypoints_flat, confidence, input_features)
    norm_center, norm_scale = load_norm_stats(ckpt_dir, input_features)
    reconstruction = run_forward(
        model,
        model_input,
        args.device,
        chunk_size=chunk_size,
        stride=stride,
        norm_center=norm_center,
        norm_scale=norm_scale,
    )
    recon_kp, recon_conf = split_model_output(reconstruction, input_features)
    if recon_conf is None:
        recon_conf = np.ones_like(confidence, dtype=np.float32)

    # Undo the causal latency budget. Trained with causal_loss_shift=N, the
    # model emits at position t the reconstruction of gt[t-N], so recon trails
    # the input (and the audio) by N frames. Advance recon by N to realign to
    # the GT timeline (repeat the last frame to keep length), matching the
    # de-shift in train_confidence.py's eval.
    deshift = args.deshift
    if deshift < 0:
        deshift = int(getattr(cfg, "causal_loss_shift", 0) or 0)
    if deshift > 0 and recon_kp.shape[0] > deshift:
        pad_kp = np.repeat(recon_kp[-1:], deshift, axis=0)
        recon_kp = np.concatenate([recon_kp[deshift:], pad_kp], axis=0)
        pad_c = np.repeat(recon_conf[-1:], deshift, axis=0)
        recon_conf = np.concatenate([recon_conf[deshift:], pad_c], axis=0)
    print(f"De-shift applied: {deshift} frame(s)", flush=True)

    # Causal tokenizers with temporal downsampling can emit one extra/fewer
    # frame than the input; align on the shorter length before comparing.
    T_cmp = min(keypoints_flat.shape[0], recon_kp.shape[0])
    coord_l1 = float(np.abs(keypoints_flat[:T_cmp] - recon_kp[:T_cmp]).mean())
    print(f"Coordinate reconstruction L1: {coord_l1:.6f}", flush=True)

    out_path = args.out or default_output_path(input_path, model_name)
    token_path = save_encoded_tokens(
        model,
        model_input,
        args.device,
        out_path,
        norm_center=norm_center,
        norm_scale=norm_scale,
        checkpoint=model_name,
        pose_path=pose_path,
        target_fps=target_fps,
    )
    print(f"Saved tokens -> {token_path}", flush=True)

    src_video = video_path if video_path and video_path.exists() else None
    overlay_path = args.overlay_out or out_path.with_name(f"{out_path.stem}_overlay{out_path.suffix}")

    print(f"Rendering comparison -> {out_path}", flush=True)
    render_comparison_video(
        orig_kp=keypoints_flat,
        recon_kp=recon_kp,
        global_state=global_state,
        confidence=confidence,
        recon_confidence=recon_conf,
        out_path=out_path,
        video_path=src_video,
        fps=target_fps,
        max_seconds=args.max_seconds,
        conf_thresh=args.conf_thresh,
        conf_lo=args.conf_lo,
        conf_hi=args.conf_hi,
    )
    print(f"Rendering overlay -> {overlay_path}", flush=True)
    render_overlay_video(
        orig_kp=keypoints_flat,
        recon_kp=recon_kp,
        global_state=global_state,
        out_path=overlay_path,
        video_path=src_video,
        fps=target_fps,
        max_seconds=args.max_seconds,
        conf_lo=args.conf_lo,
        conf_hi=args.conf_hi,
    )
    print(f"Done -> {out_path} , {overlay_path}", flush=True)


if __name__ == "__main__":
    main()
