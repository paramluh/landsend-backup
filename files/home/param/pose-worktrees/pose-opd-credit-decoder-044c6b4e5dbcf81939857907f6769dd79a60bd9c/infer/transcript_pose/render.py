"""Exact production pose-tokenizer/Electron bridge plus a skeleton preview."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch


@torch.no_grad()
def render_outputs(args, codes, frame0):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
    import posetok as P
    from safetensors.torch import save_file
    codes = torch.as_tensor(codes, dtype=torch.long)
    if codes.shape[1] % 2:
        raise ValueError("An even number of media frames is required; refusing to trim pose while preserving audio")
    meta = {}
    if args.pose_latents:
        lat = P.tokens_to_latents(codes, torch.from_numpy(frame0))
        assert int(lat["valid_frames"]) == 2 * codes.shape[1]
        save_file(lat, str(args.out / "pose_latents_electron_v3.safetensors"))
        meta["pose_latents"] = dict(shape=list(lat["latents"].shape), valid_frames=int(lat["valid_frames"]),
                                    n_frames=int(lat["n_frames"]), final_padding="repeat last pose frame once")
    if args.skeleton_video:
        # This is the same decode/denormalize/render sequence as tokens_to_latents.
        tok = P._M.get("tok")
        if tok is None:
            folder = os.environ.get("POSE_TOKENIZER_DIR") or P.snapshot_download(P.TOKENIZER[0], revision=P.TOKENIZER[1])
            tok = P.PoseTokenizer.from_pretrained(folder, device="cuda")
        kp0 = torch.from_numpy(np.ascontiguousarray(frame0[:, [1, 0, 2]]))
        gs = P.global_state(kp0.numpy())
        dec = P.decode(tok, codes)
        T = len(dec)
        f = dec.view(T, 55, 3)
        g = gs.expand(T, 3)
        pos = P.offsets_to_positions(g[:, :2], f[..., :2] * g[:, None, 2:3])
        kp = torch.zeros(T, 133, 3)
        kp[:, P.SEL, :2] = torch.stack([pos[..., 1] * P.W, pos[..., 0] * P.H], -1)
        kp[:, P.SEL, 2] = f[..., 2]
        np.save(args.out / "pose_joints_xyc.npy", kp.numpy())
        recon, g = P.raw_to_recon(kp, kp0, P.H, P.W)
        canvas = P.render_pose_canvas(recon[None].cuda(), g[None].cuda(), P.H, P.W)
        rgb = ((canvas[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{P.W}x{P.H}", "-framerate", "25", "-i", "pipe:0"]
        if (args.out / "audio.wav").exists():
            cmd += ["-i", str(args.out / "audio.wav"), "-c:a", "aac", "-shortest"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.out / "skeleton.mp4")]
        subprocess.run(cmd, input=rgb.tobytes(), check=True)
        meta["skeleton"] = dict(frames=T, fps=25, width=P.W, height=P.H, coordinate_order="pose_joints_xyc.npy is x,y,confidence")
    if args.video_sample:
        package = args.package.resolve()
        pm = json.loads((package / "meta.json").read_text())
        dst = args.out / "video_sample"
        dst.mkdir()
        for name in ("sink_latent.safetensors",):
            shutil.copyfile(package / name, dst / name)
        for name in ("pose_latents_electron_v3.safetensors", "mimi.safetensors"):
            shutil.copyfile(args.out / name, dst / name)
        pm.update(frames=int(codes.shape[1]), source_package=str(package))
        (dst / "meta.json").write_text(json.dumps(pm, indent=2) + "\n")
        meta["video_sample"] = str(dst.resolve())
    return meta


def main():
    import argparse
    from safetensors.torch import load_file

    p = argparse.ArgumentParser(description="Render saved pose codes without reloading PPM")
    p.add_argument("out", type=Path, help="Existing inference output directory")
    p.add_argument("--frame0", type=Path, help="[133,3] y,x,confidence pixels on 480x256")
    p.add_argument("--package", type=Path, help="Identity package with llm_inputs.safetensors and sink_latent.safetensors")
    p.add_argument("--pose-latents", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skeleton-video", action="store_true")
    p.add_argument("--video-sample", action="store_true")
    a = p.parse_args()
    if a.video_sample and not a.package:
        p.error("--video-sample requires --package")
    if a.video_sample:
        a.pose_latents = True
    path = a.frame0 or (a.package / "llm_inputs.safetensors" if a.package else a.out / "frame0_pose.npy")
    frame0 = (load_file(str(path))["frame0_pose"].numpy() if path.suffix == ".safetensors" else np.load(path, allow_pickle=False))
    frame0 = np.asarray(frame0, np.float32)
    if frame0.shape != (133, 3) or not np.isfinite(frame0).all():
        raise ValueError("Expected finite frame0 [133,3] y,x,confidence")
    if abs(frame0[5, 1] - frame0[6, 1]) <= 0:
        raise ValueError("Frame-zero horizontal shoulder width must be positive")
    codes = load_file(str(a.out / "pose_tokens.safetensors"))["codes"]
    result = render_outputs(a, codes, frame0)
    result.update(frame0=str(path.resolve()), pose_tokenizer=P_TOKENIZER,
                  electron_encoder=os.environ.get("ELECTRON_ENCODER"))
    (a.out / "render.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


P_TOKENIZER = "InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0de89408cb054da618e309d4334596252"


if __name__ == "__main__":
    main()
