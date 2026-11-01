#!/usr/bin/env python3
"""Agent-friendly smoke test for the default 32cb tokenizer config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml

from pose_tokenizer import (
    AGENT_DEFAULT_LOCAL_CHECKPOINT,
    AGENT_DEFAULT_TOKENIZER_REPO,
    PoseTokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the agent-default 32cb tokenizer, run encode/decode on synthetic "
            "data, and assert expected model properties."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/agent_default_32cb_tokenizer.yaml"),
        help="YAML config with tokenizer_repo/local_checkpoint/expect fields.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "local", "hub"),
        default="auto",
        help="Where to load from: local checkpoint, Hub repo, or auto fallback.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for loading and inference.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=76,
        help="Number of synthetic frames for smoke-test encode/decode.",
    )
    return parser.parse_args()


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("hf_token")
        or os.environ.get("xabi_token")
        or os.environ.get("canopy_token")
    )


def _load_cfg(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_source(cfg: dict, source_mode: str) -> tuple[str, bool]:
    repo = str(cfg.get("tokenizer_repo") or AGENT_DEFAULT_TOKENIZER_REPO)
    local = str(cfg.get("local_checkpoint") or AGENT_DEFAULT_LOCAL_CHECKPOINT)
    local_exists = Path(local).is_dir()

    if source_mode == "local":
        if not local_exists:
            raise FileNotFoundError(f"Requested --source local but missing: {local}")
        return local, False
    if source_mode == "hub":
        return repo, True

    if local_exists:
        return local, False
    return repo, True


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)
    expected = cfg.get("expect", {})
    source, using_hub = _resolve_source(cfg, args.source)

    load_kwargs = {}
    if using_hub:
        token = _hf_token()
        if token:
            load_kwargs["token"] = token

    tokenizer = PoseTokenizer.from_pretrained(source, device=args.device, **load_kwargs)
    input_features = int(tokenizer.config.input_features)
    n_codebooks = int(tokenizer.config.n_codebooks)
    codebook_size = int(tokenizer.config.codebook_size)

    exp_codebooks = int(expected.get("n_codebooks", n_codebooks))
    exp_cb_size = int(expected.get("codebook_size", codebook_size))
    if n_codebooks != exp_codebooks:
        raise AssertionError(f"n_codebooks mismatch: got {n_codebooks}, expected {exp_codebooks}")
    if codebook_size != exp_cb_size:
        raise AssertionError(f"codebook_size mismatch: got {codebook_size}, expected {exp_cb_size}")

    torch.manual_seed(0)
    sample = torch.randn(1, args.frames, input_features, device=args.device)
    codes = tokenizer.encode(sample)
    recon = tokenizer.decode(codes)

    if len(codes) != n_codebooks:
        raise AssertionError(f"Encoded {len(codes)} codebooks, expected {n_codebooks}")
    if recon.shape[-1] != input_features:
        raise AssertionError(
            f"Reconstruction feature width mismatch: got {recon.shape[-1]}, expected {input_features}"
        )
    if not torch.isfinite(recon).all():
        raise AssertionError("Reconstruction contains non-finite values")

    l1 = float((sample.detach().cpu() - recon.detach().cpu()).abs().mean())
    print(
        "SMOKE TEST PASSED | "
        f"source={source} | device={args.device} | "
        f"n_codebooks={n_codebooks} | codebook_size={codebook_size} | "
        f"input_features={input_features} | recon_l1={l1:.6f}"
    )


if __name__ == "__main__":
    main()
