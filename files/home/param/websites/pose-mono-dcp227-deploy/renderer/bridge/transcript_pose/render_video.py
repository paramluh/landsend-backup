"""Run the step-2911 nine-reference video student on PPM audio and pose outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file, save_file
import yaml

MODELS = Path(os.environ.get("POSE_MODELS", "/home/param/pose-mono-dcp227/models"))


def prepare(args):
    sample, sink_path, out = args.sample.resolve(), args.sink.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError(f"Choose a new output directory: {out}")
    repo = args.video_repo.resolve()
    if not (repo / "inference/prepare_distilled.py").is_file():
        raise ValueError(f"Pull param/bimodal with the step-2911 inference entrypoint: {repo}")
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    source = {"git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
              "runner_sha256": hashlib.sha256((repo / "inference/distilled.py").read_bytes()).hexdigest()}
    for path in (args.checkpoint, args.vae, args.video_python):
        if not path.is_file():
            raise ValueError(f"Missing inference asset: {path}")
    if not args.mimi.is_dir():
        raise ValueError(f"Missing Mimi model directory: {args.mimi}")
    pose_path = sample / "pose_latents_electron_v3.safetensors"
    mimi_path = sample / "mimi.safetensors"
    pose = load_file(str(pose_path))["latents"]
    codes = load_file(str(mimi_path))["codes"]
    sink = load_file(str(sink_path))["latents"]
    if pose.ndim != 4 or tuple(pose.shape[1:]) != (48, 30, 16):
        raise ValueError(f"Expected pose latents [T,48,30,16], got {pose.shape}")
    if sink.shape != (1, 48, 30, 16):
        raise ValueError(f"Expected sink [1,48,30,16], got {sink.shape}")
    if codes.ndim != 2 or codes.shape[0] != 32 or codes.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("Expected integer Mimi codes [32,F]")
    if codes.min() < 0 or codes.max() >= 2048:
        raise ValueError("Mimi codes must be local IDs in [0,2047]")
    if not torch.isfinite(pose).all() or not torch.isfinite(sink).all():
        raise ValueError("Pose and sink latents must be finite")
    length = min(pose.shape[0], codes.shape[1] // 2)
    if args.max_latent_frames is not None:
        if not 2 <= args.max_latent_frames <= length:
            raise ValueError(f"--max-latent-frames must be in [2,{length}]")
        length = args.max_latent_frames
    if length < 2:
        raise ValueError("Need at least two usable video latent frames")
    out.mkdir(parents=True)
    clip = out / "inputs"
    clip.mkdir()
    # The legacy loader uses video_T for its length cap. Generation reads only
    # frame zero; zero placeholders supply no future appearance information.
    timeline = torch.zeros(length, 48, 30, 16, dtype=sink.dtype)
    timeline[:1] = sink
    save_file({"latents": timeline}, str(clip / "video_latents.safetensors"))
    # Latest entrypoint validates equal pose/video input lengths before applying
    # its media cap. Keep the same usable prefix, preserving the original file.
    save_file({"latents": pose[:length].contiguous()}, str(clip / "pose_latents_electron_v3.safetensors"))
    (clip / "mimi_local.safetensors").symlink_to(mimi_path)
    config = {
        "sample_dir": str(clip), "output_dir": str(out / "generation"),
        "pose_file": "pose_latents_electron_v3.safetensors", "mimi_file": "mimi_local.safetensors",
        "type": "universal_wide_bimodal", "seed": args.seed, "batch_size": 1,
        "max_latent_frames": length, "minimum_latent_frames": length, "fps": 25,
        "compile": True, "cuda_graph": True, "wavefront": True, "renoising": False,
        "runtime_audio_guidance_scale": 1.0, "vae_path": str(args.vae.resolve()),
        "refs": {"bank": str(args.reference_bank.resolve()), "actor_id": args.actor_id, "clip_key": args.clip_key},
        "stabilising": {"checkpoint_path": str(args.checkpoint.resolve()), "num_history": 3,
                        "num_present": 4, "history_t": 0.2, "substeps": 2},
        "resolving": {"checkpoint_path": str(args.checkpoint.resolve()), "num_history": 2,
                      "num_present": 1, "t_start": 0.2, "num_steps": 1},
        "audio": {"model_name": str(args.mimi.resolve()), "in_dim": 512, "num_codebooks": 16,
                  "num_layers": 4, "cross_attention_codes": 4, "drop": 2,
                  "injection_offset": 4, "injection_interval": 3, "max_blocks": 8},
    }
    config_path = out / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    receipt = {
        "source_sample": str(sample), "sink": str(sink_path), "video_repo": str(repo),
        "video_source": source, "vae": str(args.vae.resolve()),
        "mimi_source": "PPM-generated local code IDs; no waveform re-encoding",
        "input_media_frames": int(codes.shape[1]), "usable_latent_frames": int(length),
        "decoded_video_frames": int(4 * length - 3), "duration_seconds": (4 * length - 3) / 25,
        "video_conditioning": "Clean sink only; later video input slots are zero placeholders",
        "reference_bank": str(args.reference_bank.resolve()), "actor_id": args.actor_id,
        "reference_clip_key": args.clip_key, "audio_guidance": False,
    }
    if (sink_path.parent / "provenance.json").is_file():
        receipt["appearance_provenance"] = json.loads((sink_path.parent / "provenance.json").read_text())
    bank = args.reference_bank.resolve()
    receipt["reference_files"] = [
        {"property": prop, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
         "bytes": path.stat().st_size}
        for prop in range(1, 10)
        if (path := bank / args.actor_id / f"property_{prop}" / "latents.safetensors").is_file()
    ]
    provenance = bank.parent / "reference-bank-provenance.json"
    if provenance.is_file():
        receipt["reference_bank_provenance"] = json.loads(provenance.read_text())
    (out / "input.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return config_path, receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sample", type=Path, help="PPM output with Mimi codes and decoded Electron pose latents")
    p.add_argument("--sink", type=Path, required=True, help="Matching identity's sink_latent.safetensors [1,48,30,16]")
    p.add_argument("--out", type=Path, required=True, help="New directory for staged input/config and generated video")
    p.add_argument("--reference-bank", type=Path, required=True, help="Bank root containing the matching actor's property folders")
    p.add_argument("--actor-id", required=True, help="Appearance actor UUID in the reference bank")
    p.add_argument("--clip-key", required=True, help="Original appearance clip paths value, for deterministic reference selection")
    p.add_argument("--video-repo", type=Path, default=Path(os.environ.get("VIDEO_REPO", "/home/param/pose-mono-dcp227/renderer/video-models")))
    p.add_argument("--video-python", type=Path, default=Path(os.environ.get("VIDEO_PYTHON", "/home/param/pose-mono-dcp227/renderer/.venv/bin/python")))
    p.add_argument("--checkpoint", type=Path, default=MODELS / "full-gt-nine-ref-endpoint-unguided-20261020/step-002911/student.safetensors")
    p.add_argument("--mimi", type=Path, default=MODELS / "mimi")
    p.add_argument("--vae", type=Path, default=MODELS / "wan22-vae/Wan2.2_VAE.pth")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-latent-frames", type=int, help="Explicit short smoke-test cap; otherwise use the complete usable timeline")
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    config, receipt = prepare(a)
    print(json.dumps(receipt, indent=2), flush=True)
    command = [str(a.video_python.absolute()), "-u", "-m", "inference.distilled", str(config)]
    subprocess.run([*command, "--check"], cwd=a.video_repo.resolve(), check=True)
    if a.prepare_only:
        return
    subprocess.run(command, cwd=a.video_repo.resolve(), check=True)


if __name__ == "__main__":
    main()
