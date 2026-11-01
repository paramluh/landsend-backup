#!/usr/bin/env python3
"""Run RTMW-X (via rtmlib) on N random aloe v1 speaking-clips, save the raw
133-keypoint poses as safetensors, and compare frame lengths against the
corresponding stored poses in aloe/v1-poses to detect truncation.

Background (see vansh-video-processing/pose-processing/chunk_v1_poses.py):
  v1-poses are built from raw 30fps npy -> resampled to 25fps -> cut into
  77-frame windows with stride 76 (1-frame overlap). The NUMBER OF CHUNKS is
  driven by the wan22 latent chunk count, so unique coverage =
  (n_chunks-1)*76 + 77 frames at 25fps. If the real clip is longer than that,
  the tail was dropped (truncated).

This script measures the real clip length by running RTMW on the mp4 itself.
"""
from __future__ import annotations

import glob
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from rtmlib import PoseTracker, Wholebody

V1 = Path("/mnt/som-fs/hands/aloe/v1")
V1POSES = Path("/mnt/som-fs/hands/aloe/v1-poses")
OUT = Path("/home/james/pose-tokenizer/rtmw_compare")
WINDOW, STRIDE, TARGET_FPS = 77, 76, 25
N = 10
SEED = 0


def ffprobe(mp4: str) -> tuple[float, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,nb_frames,duration",
         "-of", "json", mp4],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    nb = int(s.get("nb_frames", 0) or 0)
    dur = float(s.get("duration", 0) or 0)
    return fps, nb, dur


def v1pose_info(stem: str) -> tuple[int, int, int]:
    files = sorted(V1POSES.joinpath(stem).glob("*_poses_chunk*.safetensors"))
    n_chunks = len(files)
    rows = 0
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            rows += h.get_slice("reconstruction").get_shape()[0]
    unique = (n_chunks - 1) * STRIDE + WINDOW if n_chunks else 0
    return n_chunks, rows, unique


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mp4s = glob.glob(str(V1 / "*/speaking-clips/*.mp4"))
    have = [m for m in mp4s if V1POSES.joinpath(Path(m).stem).is_dir()]
    random.seed(SEED)
    sample = random.sample(have, N)
    print(f"{len(have)} clips with poses; sampling {N} (seed={SEED})", flush=True)

    print("init RTMW-X (Wholebody, mode=balanced, device=cuda)", flush=True)
    tracker = PoseTracker(Wholebody, det_frequency=1, to_openpose=False,
                          mode="balanced", backend="onnxruntime", device="cuda")

    rows_out = []
    for i, mp4 in enumerate(sample):
        stem = Path(mp4).stem
        import cv2
        cap = cv2.VideoCapture(mp4)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        kps, scs = [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            keypoints, scores = tracker(frame)
            if keypoints is not None and len(keypoints) > 0:
                bi = 0 if len(keypoints) == 1 else int(np.argmax([s.mean() for s in scores]))
                kps.append(keypoints[bi].astype(np.float32))
                scs.append(scores[bi].astype(np.float32))
            else:
                kps.append(np.full((133, 2), np.nan, np.float32))
                scs.append(np.zeros(133, np.float32))
        cap.release()
        fps, nb, dur = ffprobe(mp4)
        T = len(kps)
        save_file(
            {"keypoints": torch.from_numpy(np.stack(kps)).contiguous(),   # (T,133,2) x,y px
             "scores": torch.from_numpy(np.stack(scs)).contiguous(),       # (T,133)
             "fps": torch.tensor(fps), "resolution": torch.tensor([w, h])},
            str(OUT / f"{stem}.safetensors"))

        n_chunks, v1_rows, v1_unique = v1pose_info(stem)
        # full clip length expressed at the 25fps pose grid
        rtmw_at25 = int(round(T / fps * TARGET_FPS))
        trunc = rtmw_at25 - v1_unique  # >0 => v1-poses missing tail frames
        rows_out.append({
            "stem": stem, "mp4_fps": round(fps, 2), "mp4_frames": T,
            "mp4_dur_s": round(T / fps, 2),
            "rtmw_at25fps": rtmw_at25,
            "v1_chunks": n_chunks, "v1_rows": v1_rows, "v1_unique25": v1_unique,
            "v1_dur_s": round(v1_unique / TARGET_FPS, 2),
            "trunc_frames25": trunc, "trunc_s": round(trunc / TARGET_FPS, 2),
        })
        print(f"[{i+1}/{N}] {stem[:46]} | rtmw {T}f@{fps:.0f}={rtmw_at25}@25 | "
              f"v1 {n_chunks}ch={v1_unique}@25 | trunc={trunc}f "
              f"({trunc/TARGET_FPS:+.2f}s)", flush=True)

    (OUT / "comparison.json").write_text(json.dumps(rows_out, indent=2))
    print("\n==== SUMMARY ====")
    hdr = f"{'stem':46} {'mp4f':>5} {'fps':>4} {'rtmw@25':>7} {'v1ch':>4} {'v1@25':>6} {'trunc_f':>7} {'trunc_s':>7}"
    print(hdr)
    for r in rows_out:
        print(f"{r['stem'][:46]:46} {r['mp4_frames']:>5} {r['mp4_fps']:>4.0f} "
              f"{r['rtmw_at25fps']:>7} {r['v1_chunks']:>4} {r['v1_unique25']:>6} "
              f"{r['trunc_frames25']:>7} {r['trunc_s']:>+7.2f}")
    tr = [r["trunc_frames25"] for r in rows_out]
    print(f"\ntrunc_frames25: min={min(tr)} max={max(tr)} mean={np.mean(tr):.1f} "
          f"| clips truncated (>0): {sum(t>0 for t in tr)}/{len(tr)}")
    print(f"saved safetensors + comparison.json -> {OUT}")


if __name__ == "__main__":
    main()
