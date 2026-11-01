import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path('/mnt/paramfs/param/outputs/renoise-ablation-20261015')
sys.path.insert(0, str(ROOT / 'code'))
from models.autoencoders.wanvae import Wan2_2_VAE


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm')
    parser.add_argument('clip', type=int)
    args = parser.parse_args()
    out = ROOT / 'evaluation/raw_vae'
    out.mkdir(exist_ok=True)
    started = time.time()
    vae_path = '/mnt/paramfs/param/outputs/bimodal-inference/wan22-5b-weights/Wan2.2_VAE.pth'
    vae = Wan2_2_VAE(vae_pth=vae_path, dtype=torch.float32, device='cuda')
    source = ROOT / args.arm / f'sample_{args.clip}_latents.safetensors'
    latents = load_file(source)['latents'].to('cuda', torch.float32)
    print(f'{args.arm}/{args.clip}: full causal decode {tuple(latents.shape)}', flush=True)
    video = vae.decode([latents])[0]
    frames = video[:, 7125:7375].clamp(-1, 1).add(1).div(2).mul(255).byte().permute(1, 2, 3, 0).cpu().numpy()
    assert frames.shape == (250, 480, 256, 3), frames.shape
    path = out / f'{args.arm}_{args.clip}_285_295.npy'
    np.save(path, frames)
    record = dict(arm=args.arm, clip=args.clip, source_latents=str(source), vae_checkpoint=vae_path,
                  full_decoded_frames=int(video.shape[1]), retained_frame_indices=[7125, 7375],
                  retained_seconds=[285, 295], fps=25, shape=list(frames.shape), dtype=str(frames.dtype),
                  production_uint8_conversion=True, full_causal_context=True, elapsed_seconds=time.time() - started,
                  output=str(path))
    (out / f'{args.arm}_{args.clip}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
