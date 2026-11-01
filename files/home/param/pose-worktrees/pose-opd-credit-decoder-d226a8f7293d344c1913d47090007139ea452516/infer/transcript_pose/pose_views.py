"""CPU-only pose previews aligned to an already completed, cropped video.

Reuse the saved production skeleton when present. Older request jobs did not
save that preview; reconstruct it with the pinned tokenizer on CPU and a NumPy
port of the production confidence rasterizer. Never decode Electron latents:
those are conditioning features, not a lossless pose representation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
POSE = HERE.parent / "engine/posetok"
DEFAULT_TOKENIZER = Path("/home/param/models/pose-tokenizer-32cb-200k-noshift")
TOKENIZER_PIN = "InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0de89408cb054da618e309d4334596252"
FPS, WIDTH, HEIGHT = 25, 256, 480
RENDER = {"line_w": 2.0, "dot_r": 2.0, "conf_min": 0.2, "conf_hi": 0.6, "fade_floor": 0.05}
COLORKEY = {"color": "0x000000", "similarity": 0.025, "blend": 0.01, "opacity": 0.90}
_TOKENIZERS = {}
_KINEMATIC = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def probe(path):
    result = run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", path], capture_output=True, text=True)
    streams = json.loads(result.stdout)["streams"]
    videos = [s for s in streams if s["codec_type"] == "video"]
    audios = [s for s in streams if s["codec_type"] == "audio"]
    if len(videos) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    v = videos[0]
    if not str(v.get("nb_frames", "")).isdigit():
        raise ValueError(f"Missing frame count: {path}")
    return {"frames": int(v["nb_frames"]), "seconds": float(v["duration"]),
            "width": int(v["width"]), "height": int(v["height"]),
            "fps": str(v["avg_frame_rate"]), "start_seconds": float(v.get("start_time", 0)),
            "audio": [{"codec": a["codec_name"], "sample_rate": int(a["sample_rate"]),
                       "seconds": float(a["duration"]), "start_seconds": float(a.get("start_time", 0))}
                      for a in audios]}


def audio_sha(path):
    """Hash compressed audio packets, proving the soundtrack was copied intact."""
    result = run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-c:a", "copy",
                  "-f", "hash", "-hash", "sha256", "-"], capture_output=True, text=True)
    return result.stdout.strip().split("=", 1)[1]


def audio_pcm_sha(path):
    """Include AAC priming/trim metadata by also hashing the decoded soundtrack."""
    result = run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-c:a", "pcm_s16le",
                  "-f", "hash", "-hash", "sha256", "-"], capture_output=True, text=True)
    return result.stdout.strip().split("=", 1)[1]


def require_timeline(info, frames, *, audio=True):
    if (info["frames"] != frames or (info["width"], info["height"]) != (WIDTH, HEIGHT)
            or Fraction(info["fps"]) != FPS or abs(info["seconds"] - frames / FPS) > .001
            or abs(info["start_seconds"]) > .001):
        raise ValueError(f"Pose/video timeline mismatch: {info}")
    if audio and (len(info["audio"]) != 1 or info["audio"][0]["sample_rate"] != 24000
                  or info["audio"][0]["codec"] != "aac" or abs(info["audio"][0]["start_seconds"]) > .001
                  or abs(info["audio"][0]["seconds"] - frames / FPS) > .05):
        raise ValueError(f"Invalid soundtrack: {info['audio']}")


def geometry():
    global _KINEMATIC
    if _KINEMATIC is None:
        spec = importlib.util.spec_from_file_location("_pose_views_kinematic", POSE / "kinematic.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _KINEMATIC = module
    return _KINEMATIC


def load_frame0(path):
    if path is None:
        raise ValueError("CPU reconstruction requires the original frame-zero pose")
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file
        value = load_file(str(path))["frame0_pose"]
    else:
        value = np.load(path, allow_pickle=False)
    value = np.asarray(value, np.float32)
    if value.shape != (133, 3) or not np.isfinite(value).all() or abs(value[5, 1] - value[6, 1]) <= 0:
        raise ValueError("Expected finite frame0 [133,3] in y,x,confidence pixel order with nonzero shoulder width")
    return value


def decode_joints_cpu(codes_path, frame0_path, tokenizer_dir=DEFAULT_TOKENIZER, threads=2):
    """Return the same x,y,confidence joint representation saved by render.py."""
    import torch
    from safetensors.numpy import load_file

    torch.set_num_threads(threads)
    codes = np.asarray(load_file(str(codes_path))["codes"])
    if codes.ndim != 2 or codes.shape[0] != 32 or codes.shape[1] % 2 or codes.shape[1] < 2:
        raise ValueError("Expected an even number of pose-token frames, shape [32,F]")
    if not np.issubdtype(codes.dtype, np.integer) or codes.min() < 0 or codes.max() >= 2048:
        raise ValueError("Pose codes must be local integer IDs in [0,2047]")
    tokenizer_dir = Path(tokenizer_dir).resolve()
    if not (tokenizer_dir / "model.safetensors").is_file():
        raise ValueError(f"Local pinned pose tokenizer is missing: {tokenizer_dir}")
    if tokenizer_dir not in _TOKENIZERS:
        # Import only the tokenizer package; posetok.__init__ also imports the
        # GPU Electron encoder, which is deliberately unnecessary here.
        sys.path.insert(0, str(POSE))
        from pose_tokenizer import PoseTokenizer
        _TOKENIZERS[tokenizer_dir] = PoseTokenizer.from_pretrained(tokenizer_dir, device="cpu")
    tok = _TOKENIZERS[tokenizer_dir]
    with torch.inference_mode():
        dec = tok.decode([torch.from_numpy(row.copy()).long()[None] for row in codes]).float()[0]
        if dec.shape != (2 * codes.shape[1], 165) or not torch.isfinite(dec).all():
            raise ValueError("Invalid CPU tokenizer output")
        K = geometry()
        frame0 = load_frame0(frame0_path)
        p = frame0[K.WHOLEBODY_INDEX]
        yx = np.stack([p[:, 0] / HEIGHT, p[:, 1] / WIDTH], -1)
        root = torch.from_numpy((yx[5] + yx[6]) / 2)
        width = float(abs(yx[5, 1] - yx[6, 1]))
        f = dec.reshape(-1, 55, 3)
        pos = K.offsets_to_positions(root.expand(len(f), 2), f[..., :2] * width)
        kp = torch.zeros(len(f), 133, 3)
        kp[:, K.WHOLEBODY_INDEX, :2] = torch.stack([pos[..., 1] * WIDTH, pos[..., 0] * HEIGHT], -1)
        kp[:, K.WHOLEBODY_INDEX, 2] = f[..., 2]
    return kp.numpy()


def raster_coordinates(joints, frame0_path):
    """Apply the exact production CPU normalization and renderer inverse."""
    import torch
    K = geometry()
    kp = torch.from_numpy(np.asarray(joints, np.float32))
    if kp.ndim != 3 or tuple(kp.shape[1:]) != (133, 3) or not torch.isfinite(kp).all():
        raise ValueError("Expected finite joints [T,133,3] in x,y,confidence order")
    frame0 = load_frame0(frame0_path)
    kp0 = torch.from_numpy(np.ascontiguousarray(frame0[:, [1, 0, 2]]))
    recon, state = K.raw_to_recon(kp, kp0, HEIGHT, WIDTH)
    feat = recon.reshape(-1, 55, 3)
    pos = K.offsets_to_positions(state[:, :2], feat[..., :2] * state[0, 2])
    return np.stack([pos[..., 1].numpy() * WIDTH, pos[..., 0].numpy() * HEIGHT,
                     feat[..., 2].numpy()], -1).astype(np.float32)


def confidence_color(confidence):
    """Exact production red→yellow→green ramp, including low-score fading."""
    c = np.float32(np.clip(confidence, 0, 1))
    t = np.float32(np.clip((c - np.float32(.2)) / (np.float32(.6) - np.float32(.2)), 0, 1))
    fade = np.float32(np.clip((c - np.float32(.05)) / np.float32(.15), 0, 1))
    red = np.float32(255) if t < .5 else np.float32(255) * (np.float32(1) - (t - np.float32(.5)) / np.float32(.5))
    green = np.float32(255) * (t / np.float32(.5)) if t < .5 else np.float32(255)
    # Mirror render.py: canvas is normalized to [-1,1], then rounded to uint8.
    color = np.array([red * fade, green * fade, 0], np.float32)
    canvas = color / np.float32(255) * np.float32(2) - np.float32(1)
    return np.rint((canvas + np.float32(1)) * np.float32(127.5)).astype(np.uint8)


def raster_frame(points):
    """NumPy equivalent of pose_raster.cu, with clipped primitive bounding boxes.

    Pixel centers are integer coordinates; lines have radius 1 and dots radius
    2. Connections then joints overwrite earlier primitives in production order.
    The source rasterizer has hard edges, not antialiasing.
    """
    K = geometry()
    connections = K.FACE_CONNECTIONS + K.BODY_CONNECTIONS + K.LEFT_HAND_CONNECTIONS + K.RIGHT_HAND_CONNECTIONS
    joints = sorted({j for pair in connections for j in pair})
    result = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    def box(x1, y1, x2, y2, radius):
        xa, xb = max(0, math.floor(min(x1, x2) - radius)), min(WIDTH - 1, math.ceil(max(x1, x2) + radius))
        ya, yb = max(0, math.floor(min(y1, y2) - radius)), min(HEIGHT - 1, math.ceil(max(y1, y2) + radius))
        if xa > xb or ya > yb:
            return None
        yy, xx = np.mgrid[ya:yb + 1, xa:xb + 1].astype(np.float32)
        return result[ya:yb + 1, xa:xb + 1], yy, xx

    for i, j in connections:
        x1, y1, ci = points[i]
        x2, y2, cj = points[j]
        confidence = min(ci, cj)
        if confidence < .05:
            continue
        target = box(x1, y1, x2, y2, 1)
        if target is None:
            continue
        canvas, yy, xx = target
        ay, ax = np.float32(y2 - y1), np.float32(x2 - x1)
        wy, wx = yy - y1, xx - x1
        length = max(np.float32(ay * ay + ax * ax), np.float32(1e-6))
        t = np.clip((wy * ay + wx * ax) / length, 0, 1)
        dy, dx = wy - t * ay, wx - t * ax
        mask = np.sqrt(dy * dy + dx * dx) <= np.float32(1)
        canvas[mask] = confidence_color(confidence)
    for j in joints:
        x, y, confidence = points[j]
        if confidence < .05:
            continue
        target = box(x, y, x, y, 2)
        if target is None:
            continue
        canvas, yy, xx = target
        dy, dx = yy - y, xx - x
        canvas[np.sqrt(dy * dy + dx * dx) <= np.float32(2)] = confidence_color(confidence)
    return result


def save_raster(joints, frame0, path, threads):
    coords = raster_coordinates(joints, frame0)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-threads", str(threads), "-movflags", "+faststart", str(path)]
    with tempfile.TemporaryFile() as errors:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for frame in coords:
                child.stdin.write(raster_frame(frame).tobytes())
            child.stdin.close()
            if child.wait() != 0:
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace"))
        except BaseException:
            if child.poll() is None:
                child.terminate()
            child.wait()
            raise


def validate_pose_views(out, video=None):
    out = Path(out).resolve()
    receipt = json.loads((out / "pose-views.json").read_text())
    if receipt.get("format") != "pose-views-v1" or receipt.get("status") != "complete":
        raise ValueError("Invalid pose-view receipt")
    video = Path(video or receipt["source_video"]["path"]).resolve()
    if sha(video) != receipt["source_video"]["sha256"]:
        raise ValueError("Original video changed after pose-view generation")
    expected_audio = audio_sha(video)
    expected_pcm = audio_pcm_sha(video)
    frames = receipt["frames"]
    require_timeline(probe(video), frames)
    if expected_audio != receipt["audio_packet_sha256"]:
        raise ValueError("Original soundtrack changed")
    if expected_pcm != receipt["audio_pcm_s16le_sha256"]:
        raise ValueError("Original decoded soundtrack changed")
    for name in ("pose.mp4", "overlay.mp4"):
        path = out / name
        item = receipt["artifacts"][name]
        if sha(path) != item["sha256"]:
            raise ValueError(f"Pose-view checksum mismatch: {path}")
        require_timeline(probe(path), frames)
        if audio_sha(path) != expected_audio:
            raise ValueError(f"Pose-view soundtrack differs: {path}")
        if audio_pcm_sha(path) != expected_pcm:
            raise ValueError(f"Pose-view decoded soundtrack differs: {path}")
    return receipt


def build_pose_views(video, pose_dir, out=None, frame0=None, tokenizer_dir=DEFAULT_TOKENIZER, threads=2):
    video, pose_dir = Path(video).resolve(), Path(pose_dir).resolve()
    out = Path(out).resolve() if out is not None else video.parent
    if not 1 <= threads <= 16:
        raise ValueError("CPU threads must be between 1 and 16")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "pose-views.json").exists():
        return validate_pose_views(out, video)
    if any((out / name).exists() for name in ("pose.mp4", "overlay.mp4")):
        raise ValueError("Pose-view outputs exist without a valid receipt; inspect before retrying")
    info = probe(video)
    frames = info["frames"]
    require_timeline(info, frames)
    source_sha, source_audio_sha, source_pcm_sha = sha(video), audio_sha(video), audio_pcm_sha(video)
    source = {"renderer": str(POSE / "raster.py"), "render_parameters": RENDER,
              "cuda_reference_sha256": sha(POSE / "csrc/pose_raster.cu"),
              "geometry_source_sha256": sha(POSE / "kinematic.py")}
    render_path = pose_dir / "render.json"
    if render_path.is_file():
        render = json.loads(render_path.read_text())
        source["render_metadata_sha256"] = sha(render_path)
        frame0 = frame0 or render.get("frame0")
    with tempfile.TemporaryDirectory(prefix=".pose-views-", dir=out) as temporary:
        temp = Path(temporary)
        raster = pose_dir / "skeleton.mp4"
        if raster.is_file():
            source.update(kind="saved-production-skeleton", path=str(raster), sha256=sha(raster),
                          note="Saved production renderer preview; H.264 compression, not the raw float Electron input.")
        else:
            joints_path = pose_dir / "pose_joints_xyc.npy"
            if joints_path.is_file():
                joints = np.load(joints_path, allow_pickle=False)
                source.update(kind="saved-joints-cpu-raster", path=str(joints_path), sha256=sha(joints_path))
            else:
                codes_path = pose_dir / "pose_tokens.safetensors"
                joints = decode_joints_cpu(codes_path, frame0, tokenizer_dir, threads)
                source.update(kind="pose-codes-cpu-decode-and-raster", path=str(codes_path), sha256=sha(codes_path),
                              tokenizer=TOKENIZER_PIN, tokenizer_weights_sha256=sha(Path(tokenizer_dir) / "model.safetensors"))
            if len(joints) < frames:
                raise ValueError("Saved pose does not cover the published video")
            raster = temp / "raster.mp4"
            save_raster(joints[:frames], frame0, raster, threads)
            source.update(frame0=str(Path(frame0).resolve()), frame0_sha256=sha(frame0),
                          note="Production topology, normalization, confidence colors and primitive widths on CPU; floating-point boundary differences from CUDA are possible.")
        raster_info = probe(raster)
        if (raster_info["frames"] < frames or Fraction(raster_info["fps"]) != FPS
                or (raster_info["width"], raster_info["height"]) != (WIDTH, HEIGHT)
                or abs(raster_info["start_seconds"]) > .001):
            raise ValueError("Raster cannot be aligned to the published video without a time/size transform")
        source["raster_frames_available"] = raster_info["frames"]
        encoding = ["-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-threads", str(threads), "-c:a", "copy", "-movflags", "+faststart"]
        common = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", str(threads), "-i", raster,
                  "-threads", str(threads), "-i", video]
        run([*common, "-map", "0:v:0", "-map", "1:a:0", "-vf",
             f"trim=end_frame={frames},setpts=PTS-STARTPTS", *encoding, temp / "pose.mp4"])
        graph = (f"[0:v]trim=end_frame={frames},setpts=PTS-STARTPTS,format=rgb24,"
                 "colorkey=0x000000:0.025:0.01,format=rgba,colorchannelmixer=aa=0.90[pose];"
                 "[1:v]setpts=PTS-STARTPTS[base];[base][pose]overlay=shortest=1:format=auto[v]")
        run([*common, "-filter_complex_threads", str(threads), "-filter_complex", graph,
             "-map", "[v]", "-map", "1:a:0", *encoding, temp / "overlay.mp4"])
        artifacts = {}
        for name in ("pose.mp4", "overlay.mp4"):
            path = temp / name
            output_info = probe(path)
            require_timeline(output_info, frames)
            if audio_sha(path) != source_audio_sha:
                raise ValueError(f"Copied soundtrack differs in {name}")
            if audio_pcm_sha(path) != source_pcm_sha:
                raise ValueError(f"Copied decoded soundtrack differs in {name}")
            run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", str(threads), "-i", path,
                 "-f", "null", "-"], stdout=subprocess.DEVNULL)
            artifacts[name] = {"sha256": sha(path), "bytes": path.stat().st_size, **output_info,
                               "full_cpu_decode_passed": True}
        if sha(video) != source_sha:
            raise ValueError("Source video changed while making pose views")
        receipt = {"format": "pose-views-v1", "status": "complete", "cpu_only": True,
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "source_video": {"path": str(video), "sha256": source_sha}, "source_pose": source,
                   "frames": frames, "seconds": frames / FPS, "fps": FPS, "width": WIDTH, "height": HEIGHT,
                   "alignment": {"first_pose_frame": 0, "first_video_frame": 0, "temporal_shift_frames": 0,
                                 "operation": "Keep the same prefix of the saved pose timeline as the final published video; no resampling, padding, looping or motion retargeting."},
                   "overlay": COLORKEY, "audio_packet_sha256": source_audio_sha,
                   "audio_pcm_s16le_sha256": source_pcm_sha,
                   "audio_policy": "Unchanged AAC packets copied from the final published video", "artifacts": artifacts}
        for name in artifacts:
            (temp / name).replace(out / name)
        pending = temp / "pose-views.json"
        pending.write_text(json.dumps(receipt, indent=2) + "\n")
        pending.replace(out / "pose-views.json")
    return validate_pose_views(out, video)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--frame0", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = (validate_pose_views(args.out or args.video.parent, args.video) if args.check else
              build_pose_views(args.video, args.pose_dir, args.out, args.frame0, args.tokenizer_dir, args.threads))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
