"""
REFS_DIR/refs_<n>.safetensors (pick.py: the 9 ref latents [C, 9, h, w] + present the sampler got) ->
REFS_DIR/sample_<n>_refs.png, the 3 x 3 grid label.py's refs panel shows, each ref Wan2.2-VAE-decoded as a 1-frame
still at native size, property order, a missing property black.
  python image.py REFS_DIR   (one GPU)
"""
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from models.autoencoders.wanvae import Wan2_2_VAE  # noqa: E402

vae = Wan2_2_VAE(vae_pth=str(REPO / 'wan22-5b-weights/Wan2.2_VAE.pth'), dtype=torch.float32, device='cuda')
for path in sorted(Path(sys.argv[1]).glob('refs_*.safetensors')):
    refs = load_file(path)
    latents, present = refs['latents'], refs['present'].bool()
    grid = None
    for slot in range(9):
        if not present[slot]:
            continue
        with torch.no_grad():
            frame = vae.decode([latents[:, slot:slot + 1].cuda()])[0][:, 0]
        image = Image.fromarray(frame.clamp(-1, 1).add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
        grid = grid or Image.new('RGB', (3 * image.width, 3 * image.height), (0, 0, 0))
        grid.paste(image, ((slot % 3) * image.width, (slot // 3) * image.height))
    grid.save(path.parent / f"sample_{path.stem.split('_')[1]}_refs.png")
    print(path.name, present.int().tolist(), flush=True)
