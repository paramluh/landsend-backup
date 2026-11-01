# VENDORED for aloe-pose-monoplex-infer. Upstream: ppm/aloe-m1vox-ft/code/vendor/pinned_fa3.py. Edits (behaviour-neutral for this model): FA3 root = $FA3_DIR or a pinned HF snapshot (was a hard-coded local path).
"""Pinned FA3 attention backend for Torch 2.8 + CUDA 12.8 on Hopper."""

import importlib.util
import os
import sys

import torch
from huggingface_hub import snapshot_download
from transformers.modeling_utils import AttentionInterface

REPO_ID = "kernels-community/vllm-flash-attn3"
REVISION = "905da40048cf96bb967d35c5bea4bf38a715ace0"
BUILD = "torch28-cxx11-cu128-x86_64-linux"
IMPLEMENTATION = "pinned_fa3"

_module = None


def _load_module():
    global _module
    if _module is not None:
        return _module

    # E25 run-local copy: the cached snapshot path directly - one of 112 ranks hung inside snapshot_download's cache
    # lookup (file lock on the shared HF cache) on two launches in a row; no hub call at all here.
    root = os.environ.get("FA3_DIR")                           # a local snapshot of REPO_ID @ REVISION (holds build/<BUILD>/)
    if not root:
        root = snapshot_download(REPO_ID, revision=REVISION, allow_patterns=[f"build/{BUILD}/*"])
    assert os.path.isdir(os.path.join(root, "build", BUILD)), (root, BUILD)
    package = os.path.join(root, "build", BUILD)
    spec = importlib.util.spec_from_file_location(
        "_text_tts_fa3",
        os.path.join(package, "__init__.py"),
        submodule_search_locations=[package],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned FA3 package from {package}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _module = module
    return module


from fa3_op import fa3_func  # noqa: E402  speedopt B: the FA3 custom-op path (fa3_op.py)


def pinned_flash_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    sliding_window=None,
    softcap=None,
    is_causal=None,
    **kwargs,
):
    if dropout:
        raise ValueError("Pinned FA3 training requires attention_dropout=0")

    # Transformers attention modules use [B, H, T, D]; FA3 uses [B, T, H, D].
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if query.dtype == torch.float32:
        query = query.to(torch.get_autocast_dtype("cuda"))
        key = key.to(query.dtype)
        value = value.to(query.dtype)

    causal = is_causal if is_causal is not None else module.is_causal
    window = (
        (sliding_window - 1, sliding_window - 1)
        if sliding_window is not None
        else (-1, -1)
    )
    assert window == (-1, -1) and not softcap, (window, softcap)             # speedopt B: the custom-op path only
    output = fa3_func(query, key, value, softmax_scale=scaling, causal=causal)   # covers dense (causal) attention
    return output, None


AttentionInterface.register(IMPLEMENTATION, pinned_flash_attention_forward)
