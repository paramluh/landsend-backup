#!/usr/bin/env python3
"""Encode explicitly selected reference images and produce visual QA artifacts.

This creates a fresh bank, not a reproduction of a historical encoded bank.
Crop plans use the video repo's existing reference preparation implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from safetensors.torch import load_file, save_file

VAE_SHA256 = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb(tensor):
    pixels = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    return Image.fromarray(pixels[:, 0].permute(1, 2, 0).numpy())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--video-repo", type=Path, default=Path("/home/param/vansh-video-models"))
    parser.add_argument("--vae", type=Path, default=Path("/home/param/models/wan22-vae/Wan2.2_VAE.pth"))
    args = parser.parse_args()
    if sha256(args.vae) != VAE_SHA256:
        raise ValueError("Wan VAE checksum mismatch")
    sys.path.insert(0, str(args.video_repo))
    sys.path.insert(0, str(args.video_repo / "scripts/reference_bank_rebuild"))
    from models.autoencoders.wanvae import Wan2_2_VAE
    from scripts.reference_bank_rebuild.encode_bank import prepare

    rows = json.loads(args.selection.read_text())
    groups = {}
    for row in rows:
        if sha256(row["image_path"]) != row["source_sha256"]:
            raise ValueError(f"Source image checksum mismatch: {row['image_path']}")
        groups.setdefault(row["clip_id"], []).append(row)
    torch.set_num_threads(2)
    vae = Wan2_2_VAE(vae_pth=str(args.vae), dtype=torch.bfloat16, device="cuda")
    fingerprint = dict(vae_sha256=VAE_SHA256, torch=str(torch.__version__),
                       encoder_sha256=sha256(args.video_repo / "models/autoencoders/wanvae.py"),
                       prepare_sha256=sha256(args.video_repo / "scripts/reference_bank_rebuild/encode_bank.py"),
                       input_range=[-1, 1], latent_normalization="Wan2.2 channel mean/std", dtype="bfloat16",
                       provenance="Fresh same-track HF property images; not historical strict-bank byte identity")
    with torch.inference_mode():
        for clip_id, selected in groups.items():
            selected.sort(key=lambda row: row["property"])
            if [row["property"] for row in selected] != list(range(1, 10)):
                raise ValueError(f"Expected nine distinct property selections for {clip_id}")
            clip_root = Path(selected[0]["bank_root"]).parent
            checks = []
            previews = []
            for row in selected:
                prepared = prepare(row, args.selection.parent)
                latent = vae.encode([prepared.cuda()])[0]
                if latent.shape != (48, 1, 30, 16) or not torch.isfinite(latent).all():
                    raise ValueError(f"Invalid encoded reference: {row['image_path']}")
                value = latent.permute(1, 0, 2, 3).to(torch.bfloat16).cpu().contiguous()
                destination = Path(row["bank_root"]) / row["actor_id"] / f"property_{row['property']}" / "latents.safetensors"
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                save_file({"video": value}, str(temporary))
                temporary.replace(destination)
                preview = rgb(prepared)
                preview.save(clip_root / "reference-images" / f"encoded-input-property-{row['property']}.png")
                previews.append(preview)
                receipt = dict(row, latent_sha256=sha256(destination), latent_shape=list(value.shape))
                if row["property"] == 1:
                    reconstruction = vae.decode([value.permute(1, 0, 2, 3).cuda()])[0].cpu()
                    mse = float(((reconstruction - prepared) ** 2).mean())
                    receipt["roundtrip_psnr_db"] = float(10 * np.log10(4 / max(mse, 1e-12)))
                    if receipt["roundtrip_psnr_db"] < 25:
                        raise ValueError(f"Reference roundtrip failed: {clip_id}: {receipt['roundtrip_psnr_db']}")
                    rgb(reconstruction).save(clip_root / "reference-roundtrip.png")
                checks.append(receipt)
            sink = load_file(str(clip_root / "sink_latent.safetensors"))["latents"]
            if sink.shape != (1, 48, 30, 16) or not torch.isfinite(sink).all():
                raise ValueError(f"Invalid sink: {clip_id}")
            sink_image = rgb(vae.decode([sink.permute(1, 0, 2, 3).cuda()])[0])
            sink_image.save(clip_root / "sink-preview.png")
            sheet = Image.new("RGB", (256 * 5, 510 * 2), "white")
            draw = ImageDraw.Draw(sheet)
            for i, im in enumerate([sink_image, *previews]):
                x, y = (i % 5) * 256, (i // 5) * 510
                sheet.paste(im, (x, y + 30))
                draw.text((x + 5, y + 8), f"{clip_id} " + ("sink" if i == 0 else f"property {i}"), fill="black")
            sheet.save(clip_root / "appearance-qa.jpg", quality=92)
            receipt = dict(fingerprint, clip_id=clip_id, references=checks, reference_count=9,
                           visual_identity_review="pending", selection_sha256=sha256(args.selection))
            (clip_root / "reference-bank-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(dict(clip_id=clip_id, encoded=9, roundtrip_psnr_db=checks[0]["roundtrip_psnr_db"])), flush=True)


if __name__ == "__main__":
    main()
