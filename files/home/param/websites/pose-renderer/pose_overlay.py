#!/usr/bin/env python3
"""Cache browser overlay joints from the exact saved generation pose tokens.

CPU only. The avatar path reuses the production pose_views decoder and raster
coordinate transform. It adds precisely render_queue's boundary continuation,
then crops to the original generated video timeline. No pose inference or video
render is repeated. The original media files are never changed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SITE = Path("/home/param/websites/pose-playground")
DEFAULT_PEOPLE = Path("/home/param/datasets/pose-demo/prompts")
DEFAULT_TOKENIZER = Path("/home/param/models/pose-tokenizer-32cb-200k-noshift")
VERSION = 1
_HELPER = None


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(data)
    temporary.replace(path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode())


def helper():
    global _HELPER
    if _HELPER is None:
        path = HERE / "bridge/transcript_pose/pose_views.py"
        spec = importlib.util.spec_from_file_location("_playground_overlay_pose_views", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _HELPER = module
    return _HELPER


def stamp(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_info(run, site, people):
    request = read_json(run / "request.json", {})
    summary = read_json(run / "summary.json", {})
    person = (summary.get("person") or request.get("person")
              or (request.get("prefill") or {}).get("person") or "heldout_02")
    if not isinstance(person, str) or "/" in person or ".." in person:
        raise ValueError("Invalid actor identity")
    assets_file = people / person / "video_assets.json"
    assets = read_json(assets_file)
    source = {"tokens": stamp(run / "tokens.npz"), "person": person}
    if assets:
        frame0 = Path(assets["frame0_pose"])
        source.update(kind="avatar", frame0=stamp(frame0), assets=stamp(assets_file),
                      geometry="production frame-zero root and horizontal shoulder width")
        return source, frame0, None

    # This is render_queue's raw-only stick-video path. Its width differs from
    # avatar video, but normalized coordinates make the browser mapping exact.
    for base in (site / "openings", site / "openings-raw", site / "people_extra"):
        meta_path = base / person / "meta.json"
        meta = read_json(meta_path, {})
        fields = meta.get("sources", {}).get("pose_tokens", {}).get("frame0", {}).get("fields")
        if fields:
            gs = [*fields["frame0_root"], fields["frame0_width"]]
            if len(gs) != 3 or not np.isfinite(gs).all() or gs[2] <= 0:
                raise ValueError("Invalid raw actor frame-zero geometry")
            source.update(kind="stick", geometry=stamp(meta_path), global_state=gs)
            return source, None, gs
    raise ValueError(f"No verified pose geometry for {person}")


def export_overlay(run, *, site=DEFAULT_SITE, people=DEFAULT_PEOPLE,
                   tokenizer=DEFAULT_TOKENIZER, threads=2, force=False):
    run, site, people, tokenizer = map(Path, (run, site, people, tokenizer))
    out = run / "pose_overlay.json"
    status_path = run / "pose_overlay.status.json"
    source, frame0, gs = source_info(run, site, people)
    signature = hashlib.sha256(json.dumps({"version": VERSION, "source": source,
                                          "tokenizer": str(tokenizer.resolve())}, sort_keys=True).encode()).hexdigest()
    old = read_json(status_path, {})
    if not force and old.get("state") == "ready" and old.get("signature") == signature and out.is_file():
        return old
    atomic_json(status_path, {"state": "building", "at": time.time(), "signature": signature})
    try:
        with np.load(run / "tokens.npz", allow_pickle=False) as saved:
            codes = np.asarray(saved["pose"])
        if (codes.ndim != 2 or codes.shape[1] != 32 or len(codes) < 2
                or not np.issubdtype(codes.dtype, np.integer)
                or codes.min() < 0 or codes.max() >= 2048):
            raise ValueError("Expected local integer pose codes [F,32] in [0,2047]")
        n = len(codes)
        production = helper()
        kinematic = production.geometry()
        edges = (kinematic.FACE_CONNECTIONS + kinematic.BODY_CONNECTIONS
                 + kinematic.LEFT_HAND_CONNECTIONS + kinematic.RIGHT_HAND_CONNECTIONS)
        padding = 0
        if frame0 is not None:
            # The avatar renderer supplies these extra tail codes to the causal
            # tokenizer/Electron path, then crops the finished video to 2*n.
            padding = 2 + n % 2
            padded = np.concatenate((codes, np.repeat(codes[-1:], padding, axis=0)))
            from safetensors.numpy import save_file
            with tempfile.TemporaryDirectory(prefix="pose-overlay-") as temporary:
                path = Path(temporary) / "pose_tokens.safetensors"
                save_file({"codes": np.ascontiguousarray(padded.T, dtype=np.int64)}, str(path))
                joints = production.decode_joints_cpu(path, frame0, tokenizer, threads=threads)
            points = production.raster_coordinates(joints, frame0)[:2 * n]
            points[..., 0] /= production.WIDTH
            points[..., 1] /= production.HEIGHT
            width, height = production.WIDTH, production.HEIGHT
        else:
            import torch
            torch.set_num_threads(threads)
            os.environ.setdefault("POSE_TOKENIZER_DIR", str(tokenizer))
            if str(site) not in sys.path:
                sys.path.insert(0, str(site))
            import stick
            pos, confidence = stick.decode_pose(np.ascontiguousarray(codes.T, dtype=np.int64), gs)
            points = np.stack((pos[..., 1], pos[..., 0], confidence), axis=-1)
            width, height = stick.W, stick.H
            edges = [edge for group in stick.POSE_EDGES.values() for edge in group]
        if points.shape != (2 * n, 55, 3) or not np.isfinite(points).all():
            raise ValueError(f"Invalid decoded overlay shape/values: {points.shape}")
        points[..., 2] = np.clip(points[..., 2], 0, 1)
        value = {"version": VERSION, "run_id": run.name, "fps": 25, "width": width, "height": height,
                 "start_time": 0, "time_offset": 0, "frame_count": len(points), "duration": n / 12.5,
                 "coordinate_space": "normalized_xy", "joint_count": 55,
                 "frames": np.round(points.astype(np.float64), 6).tolist(), "edges": edges,
                 "confidence": {"min": .05, "low": .2, "high": .6},
                 "source": dict(source, boundary_padding_code_frames=padding,
                                tokenizer=production.TOKENIZER_PIN,
                                note="Actual generated pose, aligned to the avatar conditioning canvas and original video timeline")}
        encoded = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
        compressed = run / "pose_overlay.json.gz"
        atomic_bytes(compressed, gzip.compress(encoded, compresslevel=6, mtime=0))
        atomic_bytes(out, encoded)
        # Both files describe the same generation. Matching timestamps also
        # lets the API reject an older compressed sibling during publication.
        written = out.stat().st_mtime_ns
        os.utime(compressed, ns=(written, written))
        status = {"state": "ready", "at": time.time(), "signature": signature, "version": VERSION,
                  "frames": len(points), "fps": 25, "width": width, "height": height,
                  "bytes": len(encoded), "source_kind": source["kind"], "person": source["person"]}
        atomic_json(status_path, status)
        print(json.dumps({"run_id": run.name, **status}), flush=True)
        return status
    except Exception as exc:
        atomic_json(status_path, {"state": "error", "at": time.time(), "signature": signature,
                                  "error": f"{type(exc).__name__}: {exc}"[:500]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--people", type=Path, default=DEFAULT_PEOPLE)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--run", type=Path, help="Export one existing generation")
    parser.add_argument("--watch", action="store_true", help="Keep the CPU tokenizer warm and export completed generations")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.run and not args.watch:
        parser.error("Pass --run or --watch")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    options = dict(site=args.site, people=args.people, tokenizer=args.tokenizer, threads=args.threads)
    if args.run:
        export_overlay(args.run, force=args.force, **options)
        return
    failures = {}
    print("POSE_OVERLAY_WATCHER_READY", flush=True)
    while True:
        # Newest user work first; existing generations are backfilled using the
        # same CPU process. Never block or restart an inference/video worker.
        runs = sorted((args.site / "runs").iterdir(), key=lambda p: p.name, reverse=True)
        runs.sort(key=lambda p: bool(read_json(p / "request.json", {}).get("hidden")))
        for run in runs:
            tokens = run / "tokens.npz"
            if not run.is_dir() or not tokens.is_file():
                continue
            if read_json(run / "status.json", {}).get("state") not in ("done", "error", "failed", "cancelled"):
                continue
            if time.monotonic() - failures.get(run.name, -1000) < 30:
                continue
            try:
                export_overlay(run, **options)
                failures.pop(run.name, None)
            except Exception as exc:
                failures[run.name] = time.monotonic()
                # source_info can fail before export enters its decoding block.
                atomic_json(run / "pose_overlay.status.json", {"state": "error", "at": time.time(),
                                                               "error": f"{type(exc).__name__}: {exc}"[:500]})
                print(f"POSE_OVERLAY_ERROR {run.name}: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
