"""Empirically measure the encoder->VQ->decoder receptive field.

(Ported from the vansh-causal branch and extended to compare encoder_dilation
modes.) Feed a long random input, take a middle output frame as a scalar loss,
backprop, and count which input frames received non-zero gradient. The VQ
straight-through estimator passes identity gradient, so autograd traces the
full enc->dec path.
"""

import argparse

import torch

from pose_tokenizer import PoseTokenizerConfig, PoseTokenizerModel


def measure(dilation_mode: str, T: int, t_probe: int, device: str) -> None:
    cfg = PoseTokenizerConfig(
        num_keypoints=55, keypoint_dim=2, input_features=165,
        first_channel_size=512, latent_dim=128, num_blocks=3,
        downsampling_rates=[1, 2, 2], kernel_size=[3, 3, 3],
        num_residual_blocks=2, residual_units_per_block=2,
        d_model=2048, num_groups=False,
        n_codebooks=24, codebook_size=1024, codebook_dim=16, vq_strides=[1] * 24,
        normalize_codes=True, causal=True,
        encoder_dilation=dilation_mode,
    )
    model = PoseTokenizerModel(config=cfg).to(device).eval()

    x = torch.randn(1, T, cfg.input_features, device=device, requires_grad=True)
    out = model(x)
    out["reconstruction"][0, t_probe, :].sum().backward()

    g = x.grad[0].abs().sum(dim=-1)
    nz = (g > 0).nonzero().squeeze(-1)
    first, last = nz[0].item(), nz[-1].item()
    print(f"[dilation={dilation_mode:>7}] RF span = {last - first + 1:>3} frames "
          f"| left of probe = {t_probe - first:>3} | right = {last - t_probe:>2} "
          f"| frames {first}..{last}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modes", default="none,shallow,deep")
    p.add_argument("--T", type=int, default=800)
    p.add_argument("--t-probe", type=int, default=600)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    print(f"T={args.T}, probe at t={args.t_probe} (causal -> RF should be left of probe)\n")
    for m in args.modes.split(","):
        measure(m.strip(), args.T, args.t_probe, args.device)


if __name__ == "__main__":
    main()
