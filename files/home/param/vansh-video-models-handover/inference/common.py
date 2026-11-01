import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import av
import numpy as np
import torch
from safetensors.torch import load_file

from dataloading.sources import AUDIO_CODE_QUANTUM

POSE_AUDIO_SR = 24000
CKPT_PATH = "../wan22-5b-weights"
DTYPE = torch.bfloat16
FPS = 25


def resolve_path(raw, base_dir):
    p = Path(raw)
    return str(p.resolve() if p.is_absolute() else (base_dir / p).resolve())


def find_samples(samples_dir, max_samples, sample_seed):
    """Sample numbers n of samples_<n>_low_res.safetensors in samples_dir (pretrain-ds validation naming)."""
    nums = sorted(int(p.name.split('_')[1]) for p in samples_dir.glob('samples_*_low_res.safetensors'))
    if sample_seed is not None:
        np.random.default_rng(sample_seed).shuffle(nums)
    return nums[:max_samples] if max_samples else nums


def load_sample(samples_dir, n, num_codebooks, max_frames, slide_audio=False):
    """
    samples_<n>_low_res.safetensors  key "latents"       [48, L, h, w]   frame 0 -> sink
    samples_<n>_pose.safetensors     key "pose_latents"  [48, L, h, w]   (or _pose_v2, same latents)
    samples_<n>_mimi.npy             Mimi codes          [32, ~2L]       12.5 Hz
    Returns latents, pose cut to the conditioned length, the training-shaped code table, and the conditioned codes.
    slide_audio: the caller slides the code table per window (sliding_code_table), so the length is not capped
    at the table's AUDIO_CODE_QUANTUM // 2 frames.
    """
    latents = load_file(str(samples_dir / f'samples_{n}_low_res.safetensors'))['latents']
    pose_path = next(p for p in (samples_dir / f'samples_{n}_pose{v}.safetensors' for v in ('', '_v2')) if p.exists())
    pose = load_file(str(pose_path))['pose_latents']
    codes = torch.from_numpy(np.load(samples_dir / f'samples_{n}_mimi.npy')).long()
    if codes.shape[0] < num_codebooks:
        raise RuntimeError(f'samples_{n}: {codes.shape[0]} codebooks < {num_codebooks}')
    # Conditioned length: latent frames that have pose and both of their 12.5 Hz codes, inside the code table.
    L = min(latents.shape[1], pose.shape[1], codes.shape[1] // 2)
    if not slide_audio:
        L = min(L, AUDIO_CODE_QUANTUM // 2)
    # Same fixed-size table as training. The audio path is causal (token i reads codes <= (i + drop) / 2), so
    # the zero tail never reaches a generated frame, but frame f gathers tokens up to 4f - 1, i.e. code 2f: the
    # table must extend at least one code past the last conditioned frame.
    table = torch.zeros(1, codes.shape[0], AUDIO_CODE_QUANTUM, dtype=torch.long)
    length = min(codes.shape[1], AUDIO_CODE_QUANTUM)
    table[0, :, :length] = codes[:, :length]
    frames = min(L, max_frames or L)
    return latents[:, :frames].float(), pose[:, :frames].float(), table, codes[:, :2 * L]


def sliding_code_table(code_table, codes, f_last):
    """
    Training's audio window (dataloading/ds_aloe.py load_window): a window whose last real frame is f_last reads
    codes from 2a, a = max(0, f_last - (AUDIO_CODE_QUANTUM // 2 - 1)), with real frame indices shifted down by a.
    Returns (table [1, codebooks, AUDIO_CODE_QUANTUM], a); a = 0 is the fixed table.
    """
    a = max(0, f_last - (AUDIO_CODE_QUANTUM // 2 - 1))
    if a == 0:
        return code_table, 0
    table = torch.zeros_like(code_table)
    part = codes[:, 2 * a:2 * a + AUDIO_CODE_QUANTUM]
    table[0, :, :part.shape[1]] = part
    return table, a


def run_parallel(script, config_arg, num_samples):
    """One worker process per visible GPU, each running `script` on a shard of the samples."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpus = visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
    n = min(len(gpus), num_samples)
    procs = [
        subprocess.Popen(
            [sys.executable, script, config_arg, '--shard', str(i), '--num-shards', str(n)],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i]),
        )
        for i in range(n)
    ]
    if any(p.wait() for p in procs):
        raise SystemExit('a worker failed')


def save_with_audio(video_path, audio_np, tag=""):
    p = Path(video_path)
    audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=p.suffix, delete=False, dir=p.parent)
    try:
        with wave.open(tmp_wav.name, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(POSE_AUDIO_SR)
            wf.writeframes(audio_int16.tobytes())

        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", tmp_wav.name,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", tmp_out.name,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        os.replace(tmp_out.name, video_path)
        print(f"[{tag}] Video+audio saved to {video_path}")
    finally:
        for f in [tmp_wav.name, tmp_out.name]:
            if os.path.exists(f):
                os.unlink(f)
    return video_path


def _save_tensor_as_video(tensor, path, fps):
    """
    tensor: [3, T, H, W] in [-1, 1]. Saves H.264 mp4 via PyAV.
    """
    v = tensor.clamp(-1, 1).add(1).div(2).mul(255).byte()
    v = v.permute(1, 2, 3, 0).cpu().numpy()
    T, H, W, _ = v.shape
    container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    for i in range(T):
        frame = av.VideoFrame.from_ndarray(v[i], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def side_by_side(left_path, right_path, output_path, tag=""):
    """[left | right] video, cut to the shorter input, audio taken from the right input."""
    cmd = [
        "ffmpeg", "-y", "-i", str(left_path), "-i", str(right_path),
        "-filter_complex", "[0:v][1:v]hstack=inputs=2:shortest=1[v]", "-map", "[v]", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
        "-movflags", "+faststart", str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    print(f"[{tag}] Side-by-side saved to {output_path}")


def decode_wan22_vae(vae, latents, output_path, fps, audio_np=None, tag=""):
    print(f"[{tag}] Decoding latents with Wan2.2 VAE...")
    latent_list = [latents[i].to(vae.device, vae.dtype) for i in range(latents.shape[0])]
    with torch.no_grad():
        videos = vae.decode(latent_list)
    _save_tensor_as_video(videos[0], output_path, fps)
    print(f"[{tag}] Video saved to {output_path}")
    if audio_np is not None:
        save_with_audio(output_path, audio_np, tag=tag)
