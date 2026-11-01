"""Verification tests for the SEANet+transformer pose tokenizer backend.

Checks (no training data needed):
  1. param count of the real ~200M config (encoder ~= decoder), per-module.
  2. shape / token-rate contract: recon == input length, codes length == T/2,
     downsampling_factor == 2.
  3. bitrate: full 32cb == 4400 bps, 16-prefix == 2200 bps (matches conv A/B).
  4. codebook: 32 int code tensors in [0, 2048); prefix-decode 1..32 works.
  5. CAUSALITY (the key gate): perturbing inputs at/after frame tau must NOT
     change reconstruction[:tau] (fp-exact) nor codes covering only inputs < tau.

Run:
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/james/pose-tokenizer \
      /home/james/INFRA-interactive-face-diffusion/.venv-pose/bin/python \
      scripts/test_seanet_tokenizer.py --config configs/seanet_tf_2x_32cb_2048_200M.yaml
"""

from __future__ import annotations

import argparse
import math

import torch

from pose_tokenizer import PoseTokenizerModel
from pose_tokenizer.config import PoseTokenizerConfig, TrainConfig


def _count(m) -> int:
    return sum(p.numel() for p in m.parameters())


def test_param_count(model_kw: dict) -> None:
    print("\n=== [1] param count (real config) ===")
    cfg = PoseTokenizerConfig(**{
        **model_kw, "input_features": 165, "num_keypoints": 55, "keypoint_dim": 2,
    })
    model = PoseTokenizerModel(config=cfg)
    enc, dec, q = _count(model.encoder), _count(model.decoder), _count(model.quantizer)
    tot = _count(model)
    print(f"  encoder    : {enc/1e6:7.2f} M")
    print(f"  decoder    : {dec/1e6:7.2f} M")
    print(f"  quantizer  : {q/1e6:7.2f} M  (32 x 2048 x 16 codebooks)")
    print(f"  TOTAL      : {tot/1e6:7.2f} M")
    print(f"  enc/dec balance: {enc/dec:.3f} (want ~1.0)")
    assert 150e6 < tot < 260e6, f"total {tot/1e6:.1f}M not near 200M -- tune st_num_layers"
    print("  OK")


def _small_model() -> PoseTokenizerModel:
    cfg = PoseTokenizerConfig(
        model_type="seanet_transformer", input_features=165, num_keypoints=55,
        keypoint_dim=2, causal=True, conv_weight_norm=True, latent_dim=128,
        st_seanet_dim=128, st_n_filters=16, st_n_residual_layers=1, st_ratios=[1],
        st_d_model=64, st_num_heads=4, st_num_layers=2, st_dim_feedforward=128,
        st_downsample=2,
        n_codebooks=32, codebook_size=2048, codebook_dim=16, vq_strides=[1] * 32,
        quantizer_dropout=0.5, normalize_codes=True, dead_code_threshold=2.0,
    )
    return PoseTokenizerModel(config=cfg).eval()


def test_shapes_and_bitrate(device: str) -> None:
    print("\n=== [2,3,4] shapes / token-rate / bitrate / codebook ===")
    model = _small_model().to(device)
    B, T, F = 2, 80, 165
    x = torch.randn(B, T, F, device=device)
    with torch.no_grad():
        out = model(x)
        codes = model.tokenize(x)

    assert model.downsampling_factor == 2, model.downsampling_factor
    assert out["reconstruction"].shape == (B, T, F), out["reconstruction"].shape
    assert len(codes) == 32, len(codes)
    for c in codes:
        assert c.shape == (B, T // 2), c.shape
        assert c.dtype in (torch.long, torch.int64)
        assert int(c.min()) >= 0 and int(c.max()) < 2048
    print(f"  recon {tuple(out['reconstruction'].shape)}, codes 32 x {tuple(codes[0].shape)}  OK")

    token_rate = 25 / model.downsampling_factor
    full_bps = 32 * math.log2(2048) * token_rate
    pre_bps = 16 * math.log2(2048) * token_rate
    print(f"  token_rate={token_rate} Hz  full(32cb)={full_bps:.0f} bps  16-prefix={pre_bps:.0f} bps")
    assert abs(full_bps - 4400) < 1 and abs(pre_bps - 2200) < 1
    # prefix decode 1..32
    with torch.no_grad():
        for k in (1, 4, 16, 32):
            rec_k = model.decode(codes, n_codebooks=k)
            assert rec_k.shape == (B, T, F), (k, rec_k.shape)
    print("  prefix-decode k in {1,4,16,32} OK")


def test_causality(device: str) -> None:
    print("\n=== [5] causality perturbation ===")
    model = _small_model().to(device)
    B, T, F = 2, 80, 165
    tau = 40
    torch.manual_seed(0)
    x1 = torch.randn(B, T, F, device=device)
    x2 = x1.clone()
    x2[:, tau:] += torch.randn(B, T - tau, F, device=device)  # change only >= tau
    with torch.no_grad():
        r1 = model(x1)["reconstruction"]
        r2 = model(x2)["reconstruction"]
        c1 = model.tokenize(x1)
        c2 = model.tokenize(x2)

    max_pre = (r1[:, :tau] - r2[:, :tau]).abs().max().item()
    max_post = (r1[:, tau:] - r2[:, tau:]).abs().max().item()
    print(f"  recon[:tau] max abs diff = {max_pre:.3e} (must be ~0)")
    print(f"  recon[tau:] max abs diff = {max_post:.3e} (should be large -> sanity)")
    assert max_pre < 1e-4, f"CAUSALITY VIOLATION: recon[:tau] changed by {max_pre}"
    assert max_post > 1e-3, "perturbation had no downstream effect -- test is vacuous"

    # token p depends on inputs up to ~2p; tokens fully before tau (with margin)
    # must be bit-identical.
    safe = tau // 2 - 2
    code_diff = sum(int((a[:, :safe] != b[:, :safe]).sum()) for a, b in zip(c1, c2))
    print(f"  codes[:, :{safe}] mismatches across 32 codebooks = {code_diff} (must be 0)")
    assert code_diff == 0, "CAUSALITY VIOLATION: early tokens changed"
    print("  OK -- strictly causal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/seanet_tf_2x_32cb_2048_200M.yaml")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    _, model_kw = TrainConfig.from_yaml(a.config)
    test_param_count(model_kw)
    test_shapes_and_bitrate(device)
    test_causality(device)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
