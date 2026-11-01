#!/usr/bin/env python3
"""Generate a pose overlay video using RTMW-X via rtmlib.

Draws the same 50 upper-body + hand joints and 5 face keypoints
as the ViTPose overlay, for direct comparison.

Usage:
    uv run python scripts/rtmwx_overlay.py path/to/clip.mp4
    uv run python scripts/rtmwx_overlay.py path/to/clip.mp4 -o output.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from rtmlib import PoseTracker, Wholebody

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pose_tokenizer.data.kinematic import (
    SELECTED_INDICES,
    NUM_JOINTS,
    BODY_CONNECTIONS,
    LEFT_HAND_CONNECTIONS,
    RIGHT_HAND_CONNECTIONS,
    SKELETON_COLORS,
)

FACE_INDICES = list(range(5))
FACE_CONNECTIONS = [(0, 1), (0, 2), (1, 3), (2, 4)]
FACE_COLOR = "#ffff00"

_CONN_GROUPS = [
    (BODY_CONNECTIONS, SKELETON_COLORS["body"]),
    (LEFT_HAND_CONNECTIONS, SKELETON_COLORS["left_hand"]),
    (RIGHT_HAND_CONNECTIONS, SKELETON_COLORS["right_hand"]),
]


def overlay_skeleton(
    img: np.ndarray,
    keypoints_133: np.ndarray,
    scores_133: np.ndarray,
    conf_thresh: float = 0.3,
) -> np.ndarray:
    """Draw our standard skeleton on a frame using raw RTMW-X output.

    keypoints_133: (133, 2) in (x, y) pixel coords (rtmlib format).
    scores_133: (133,) confidence per joint.
    """
    out = img.copy()
    h, w = img.shape[:2]

    # Body + hands: select 50 joints
    sel_xy = keypoints_133[SELECTED_INDICES]  # (50, 2) as (x, y)
    sel_conf = scores_133[SELECTED_INDICES]   # (50,)

    for connections, hex_color in _CONN_GROUPS:
        rgb = tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        for i, j in connections:
            if sel_conf[i] < conf_thresh or sel_conf[j] < conf_thresh:
                continue
            pt1 = (int(sel_xy[i, 0]), int(sel_xy[i, 1]))
            pt2 = (int(sel_xy[j, 0]), int(sel_xy[j, 1]))
            cv2.line(out, pt1, pt2, rgb, 2, cv2.LINE_AA)

        joint_indices = sorted({idx for pair in connections for idx in pair})
        for idx in joint_indices:
            if sel_conf[idx] < conf_thresh:
                continue
            cv2.circle(out, (int(sel_xy[idx, 0]), int(sel_xy[idx, 1])), 3, rgb, -1, cv2.LINE_AA)

    # Face: 5 keypoints
    face_xy = keypoints_133[FACE_INDICES]   # (5, 2) as (x, y)
    face_conf = scores_133[FACE_INDICES]    # (5,)
    face_rgb = tuple(int(FACE_COLOR.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    for i, j in FACE_CONNECTIONS:
        if face_conf[i] < conf_thresh or face_conf[j] < conf_thresh:
            continue
        cv2.line(out, (int(face_xy[i, 0]), int(face_xy[i, 1])),
                 (int(face_xy[j, 0]), int(face_xy[j, 1])), face_rgb, 2, cv2.LINE_AA)
    for i in range(5):
        if face_conf[i] < conf_thresh:
            continue
        cv2.circle(out, (int(face_xy[i, 0]), int(face_xy[i, 1])), 4, face_rgb, -1, cv2.LINE_AA)

    return out


class OneEuroKeypointSmoother:
    """Confidence-aware One Euro smoother for streaming keypoint frames."""

    def __init__(
        self,
        fps: float,
        min_cutoff: float = 3.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
    ):
        self.fps = float(fps)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.prev_kp: np.ndarray | None = None
        self.prev_dx: np.ndarray | None = None
        self.prev_sc: np.ndarray | None = None

    @staticmethod
    def _alpha(t_e: float, cutoff: np.ndarray | float) -> np.ndarray | float:
        r = 2.0 * np.pi * cutoff * t_e
        return r / (r + 1.0)

    def __call__(
        self, kp: np.ndarray, sc: np.ndarray, conf_thresh: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        kp = kp.astype(np.float32, copy=False)
        sc = sc.astype(np.float32, copy=False)
        if self.prev_kp is None:
            self.prev_kp = kp.copy()
            self.prev_dx = np.zeros_like(kp)
            self.prev_sc = sc.copy()
            return kp, sc

        filled_kp = kp.copy()
        smoothed_sc = sc.copy()

        low = sc < conf_thresh
        had = self.prev_sc >= conf_thresh
        carry = low & had
        if carry.any():
            filled_kp[carry] = self.prev_kp[carry]
            smoothed_sc[carry] = self.prev_sc[carry] * 0.8

        t_e = 1.0 / self.fps
        a_d = self._alpha(t_e, self.d_cutoff)
        dx = (filled_kp - self.prev_kp) * self.fps
        dx_hat = a_d * dx + (1.0 - a_d) * self.prev_dx

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        smoothed_kp = a * filled_kp + (1.0 - a) * self.prev_kp

        self.prev_kp = smoothed_kp.copy()
        self.prev_dx = dx_hat.copy()
        self.prev_sc = smoothed_sc.copy()
        return smoothed_kp, smoothed_sc


def main():
    parser = argparse.ArgumentParser(description="RTMW-X pose overlay video")
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--mode", default="balanced", choices=["performance", "balanced", "lightweight"])
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--euro-min-cutoff", type=float, default=3.0)
    parser.add_argument("--euro-beta", type=float, default=0.5)
    parser.add_argument("--euro-d-cutoff", type=float, default=1.0)
    parser.add_argument("--det-frequency", type=int, default=1)
    parser.add_argument("--backend", default="onnxruntime")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path("pose_overlays")
    out_dir.mkdir(exist_ok=True)
    output = args.output or out_dir / f"{args.video.stem}_rtmwx_overlay.mp4"

    print(f"Initializing RTMW-X (mode={args.mode}, det_freq={args.det_frequency}, "
          f"backend={args.backend}, device={args.device})")
    wholebody = PoseTracker(
        Wholebody,
        det_frequency=args.det_frequency,
        to_openpose=False,
        mode=args.mode,
        backend=args.backend,
        device=args.device,
    )

    cap = cv2.VideoCapture(str(args.video))
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"Video: {T} frames @ {fps:.1f} fps ({w}x{h}), "
        f"One Euro min_cutoff={args.euro_min_cutoff}, beta={args.euro_beta}, "
        f"d_cutoff={args.euro_d_cutoff}"
    )

    smoother = OneEuroKeypointSmoother(
        fps=fps,
        min_cutoff=args.euro_min_cutoff,
        beta=args.euro_beta,
        d_cutoff=args.euro_d_cutoff,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    all_keypoints = []
    all_scores = []

    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, scores = wholebody(frame)

        img_show = frame.copy()
        if keypoints is not None and len(keypoints) > 0:
            if len(keypoints) == 1:
                best_kp, best_sc = keypoints[0], scores[0]
            else:
                best_idx = np.argmax([s.mean() for s in scores])
                best_kp, best_sc = keypoints[best_idx], scores[best_idx]

            best_kp, best_sc = smoother(best_kp, best_sc)
            img_show = overlay_skeleton(img_show, best_kp, best_sc, conf_thresh=args.kpt_thr)
            all_keypoints.append(best_kp.copy())
            all_scores.append(best_sc.copy())
        else:
            all_keypoints.append(np.full((133, 2), np.nan))
            all_scores.append(np.zeros(133))

        writer.write(img_show)

        fi += 1
        if fi % 50 == 0:
            print(f"  {fi}/{T} frames")

    cap.release()
    writer.release()

    pose_path = out_dir / f"{args.video.stem}_poses.npz"
    np.savez_compressed(
        pose_path,
        keypoints=np.stack(all_keypoints),   # (T, 133, 2) in (x, y) pixel coords
        scores=np.stack(all_scores),         # (T, 133)
        fps=fps,
        resolution=np.array([w, h]),
    )
    print(f"Saved poses → {pose_path}  ({fi} frames, 133 joints)")

    print(f"Processed {fi} frames, encoding with ffmpeg...")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", tmp_path,
            "-i", str(args.video),
            "-map", "0:v",
            "-map", "1:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    Path(tmp_path).unlink(missing_ok=True)
    print(f"Done → {output}")


if __name__ == "__main__":
    main()
