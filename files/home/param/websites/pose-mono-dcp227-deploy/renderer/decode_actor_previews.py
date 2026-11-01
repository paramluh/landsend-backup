"""Decode original heldout source video latents for the actor picker.

These are reconstructions of recorded video, never avatar/model generations.
The original video timing is retained and the recorded prompt Mimi audio is used.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, "/home/param/vansh-video-models")
import av
import torch
from safetensors.torch import load_file
from models.autoencoders.wanvae import Wan2_2_VAE
from inference.bounded_decode import decoded_chunks, rgb_bytes

ROOT = Path('/home/param/runs/pose-runs/pose-sites-20261030/actor-previews')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=int, nargs='*', default=list(range(1, 9)))
    ap.add_argument('--warmup', type=int, default=64)
    ap.add_argument('--seconds', type=float, default=10)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vae_path = Path('/home/param/models/wan22-vae/Wan2.2_VAE.pth')
    vae = Wan2_2_VAE(vae_pth=str(vae_path), dtype=torch.bfloat16, device='cuda')
    vae.model.to(dtype=torch.bfloat16)
    dest = ROOT / 'decoded'
    dest.mkdir(parents=True, exist_ok=True)
    for i in args.only:
        pid = f'heldout_{i:02d}'
        src = ROOT / 'source-latents' / pid
        prov = json.loads((src/'provenance.json').read_text())
        tensors = load_file(str(src/f"part_{prov['part']:06d}.safetensors"))
        assert tensors['latents'].shape[1:] == (48, 30, 16)
        start = round(prov['offset_seconds'] * 25)
        count = round(args.seconds * 25)
        # Latent 0 produces source frame 0; latent i>0 produces frames 4i-3..4i.
        first_latent = max(0, (start + 3) // 4 - args.warmup)
        final_latent = math.ceil((start + count - 1) / 4)
        z = tensors['latents'][first_latent:final_latent+1].permute(1, 0, 2, 3)[None].contiguous()
        # Cold decoding begins with one output frame for this first latent;
        # subsequent chunks regain the original temporal phase immediately.
        first_frame = 4 * first_latent
        trim = start - first_frame
        video = dest / f'{pid}-warm{args.warmup}.silent.mp4'
        elapsed = time.monotonic()
        written = 0
        cursor = 0
        with av.open(str(video), 'w', options={'movflags': '+faststart'}) as container:
            stream = container.add_stream('libx264', rate=25)
            stream.width, stream.height = 256, 480
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '18', 'preset': 'medium', 'threads': '2'}
            with torch.inference_mode():
                for chunk in decoded_chunks(vae, z):
                    frames = rgb_bytes(chunk)
                    for frame in frames:
                        if trim <= cursor < trim + count:
                            if written == 0:
                                from PIL import Image
                                Image.fromarray(frame).save(dest/f'{pid}-warm{args.warmup}.png')
                            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
                                container.mux(packet)
                            written += 1
                        cursor += 1
                for packet in stream.encode():
                    container.mux(packet)
        assert written == count, (pid, written, count)
        audio = ROOT.parent/'duplex-live/clips'/f'start_{pid}.wav'
        result = dest/f'op-openings-{pid}.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(video),'-i',str(audio),
                        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','128k',
                        '-t',str(args.seconds),'-movflags','+faststart',str(result)], check=True)
        record = dict(prov, preview=str(result), preview_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
                      source_frame_start=start, source_frame_end_exclusive=start+count,
                      first_decoded_latent=first_latent, decoded_latents=z.shape[2], warmup_latents=args.warmup,
                      frames=written, fps=25, seconds=args.seconds, audio=str(audio), decoder='Wan22 original source reconstruction',
                      vae=str(vae_path), elapsed_seconds=round(time.monotonic()-elapsed,3))
        (dest/f'{pid}.json').write_text(json.dumps(record,indent=2))
        print(json.dumps({'person':pid,'frames':written,'seconds':args.seconds,'elapsed':record['elapsed_seconds'],
                          'max_gpu_mib':torch.cuda.max_memory_allocated()//2**20}),flush=True)


if __name__ == '__main__':
    main()
