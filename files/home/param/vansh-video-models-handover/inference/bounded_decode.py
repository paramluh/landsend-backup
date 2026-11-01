"""Exact causal Wan VAE decoding with bounded RGB residency.

The original decoder already advances one latent frame at a time. This keeps
the same scale/conv2 computation and causal cache, but publishes each decoded
chunk instead of collecting the entire video on the GPU.
"""

import json
import os
from pathlib import Path
import time
import uuid

import av
import torch

from inference.common import save_with_audio
from models.autoencoders.wanvae import unpatchify


@torch.no_grad()
def decoded_chunks(vae, latents):
    """Yield [3,T,H,W] float32 chunks with the original [-1,1] clipping."""
    if latents.ndim != 5 or latents.shape[0] != 1:
        raise ValueError('Bounded decode requires exactly one [B,C,L,h,w] video')
    model = vae.model
    z = latents.to(device=vae.device, dtype=vae.dtype)
    device_type = torch.device(vae.device).type
    model.clear_cache()
    try:
        with torch.autocast(device_type, dtype=vae.dtype,
                            enabled=vae.dtype in (torch.float16, torch.bfloat16)):
            scale = vae.scale
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, model.z_dim, 1, 1, 1) + scale[0].view(1, model.z_dim, 1, 1, 1)
            else:
                z = z / scale[1] + scale[0]
            # Preserve the original whole-latent conv2 geometry and rounding.
            x = model.conv2(z)
            del z
            for index in range(x.shape[2]):
                model._conv_idx = [0]
                out = model.decoder(x[:, :, index:index + 1],
                                    feat_cache=model._feat_map,
                                    feat_idx=model._conv_idx,
                                    first_chunk=index == 0)
                yield unpatchify(out, patch_size=2).float().clamp_(-1, 1).squeeze(0)
    finally:
        model.clear_cache()


def rgb_bytes(chunk):
    # Match inference.common._save_tensor_as_video exactly before encoding.
    return chunk.clamp(-1, 1).add(1).div(2).mul(255).byte().permute(1, 2, 3, 0).cpu().numpy()


def decode_wan22_vae_bounded(vae, latents, output_path, fps, audio_np=None, tag='long'):
    output = Path(output_path)
    temporary = output.with_name(f'.{output.stem}.partial-{uuid.uuid4().hex}{output.suffix}')
    started = time.monotonic()
    frame_count = 0
    container = None
    chunks = decoded_chunks(vae, latents)
    try:
        container = av.open(str(temporary), mode='w', options={'movflags': '+faststart'})
        stream = container.add_stream('libx264', rate=int(round(fps)))
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'medium'}
        for chunk in chunks:
            frames = rgb_bytes(chunk)
            if not frame_count:
                stream.height, stream.width = frames.shape[1:3]
            for pixels in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format='rgb24')):
                    container.mux(packet)
                frame_count += 1
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        expected = 4 * latents.shape[2] - 3
        if frame_count != expected:
            raise RuntimeError(f'Decoded {frame_count} frames, expected {expected}')
        if audio_np is not None:
            save_with_audio(str(temporary), audio_np, tag=tag)
        os.replace(temporary, output)
    finally:
        chunks.close()
        if container is not None:
            container.close()
        temporary.unlink(missing_ok=True)
    record = {'path': str(output), 'decoded_frames': frame_count,
              'video_duration_seconds': frame_count / fps,
              'elapsed_seconds': time.monotonic() - started,
              'decoder': 'unchanged-causal-Wan-weights-and-cache-bounded-RGB',
              'vae_dtype': str(vae.dtype)}
    with (output.parent / 'bounded-decode.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)
