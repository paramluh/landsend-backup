"""Small dependency/kernel validation; does not load a checkpoint or train a model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--all-gpus", action="store_true")
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "finetune/xabi-main-17850/code"
sys.path.insert(0, str(CODE))
os.environ.setdefault("PPM_FA3_ROOT", str(ROOT / ".cache/kernels/vllm-flash-attn3"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache/triton"))
header_paths = f"{ROOT}/.venv/include/python3.12:{ROOT}/.venv/include"
os.environ["CPATH"] = header_paths + (":" + os.environ["CPATH"] if os.environ.get("CPATH") else "")

import numpy
import pyarrow
import torch
import transformers
import triton
import wandb
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
import model_ppm
import data_aloe_pose
import data_ppm
import fa3_op
import fused_triton
import pinned_fa3

assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
kernel = pinned_fa3._load_module()
report = {
    "host": socket.gethostname(),
    "uid": os.getuid(),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "triton": triton.__version__,
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "pyarrow": pyarrow.__version__,
    "wandb": wandb.__version__,
    "fsdp2_import": "passed",
    "original_model_and_data_imports": "passed",
    "fa3_revision": pinned_fa3.REVISION,
    "fa3_module": kernel.__file__,
    "gpu_count": torch.cuda.device_count(),
    "gpus": [],
}
for device_index in range(torch.cuda.device_count() if args.all_gpus else 1):
    torch.cuda.set_device(device_index)
    torch.manual_seed(20261028)
    device = torch.device("cuda", device_index)
    q = torch.randn(1, 64, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out = fa3_op.fa3_func(q, k, v, softmax_scale=None, causal=True)
    grad = torch.randn_like(out)
    out.backward(grad)
    actual_grads = [t.grad.detach().clone() for t in (q, k, v)]
    for tensor in (q, k, v):
        tensor.grad = None
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        ).transpose(1, 2)
    ref.backward(grad)
    torch.testing.assert_close(out, ref, atol=0.02, rtol=0.02)
    for actual, tensor in zip(actual_grads, (q, k, v)):
        torch.testing.assert_close(actual, tensor.grad, atol=0.04, rtol=0.04)

    # Exercise the packed-segment FA3 ABI separately from dense attention.
    qv = torch.randn(64, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn_like(qv, requires_grad=True)
    vv = torch.randn_like(qv, requires_grad=True)
    cu = torch.tensor([0, 24, 64], device=device, dtype=torch.int32)
    packed = fa3_op.fa3_varlen_func(qv, kv, vv, cu, cu, 40, 40, None, True)
    packed.float().square().mean().backward()
    assert bool(packed.isfinite().all())
    assert all(bool(t.grad.isfinite().all()) for t in (qv, kv, vv))

    # Compile and exercise the original trainer's custom Triton path.
    x = torch.randn(32, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(128, device=device, dtype=torch.bfloat16, requires_grad=True)
    norm = fused_triton.rms(x, weight, 1e-6)
    norm.float().square().mean().backward()
    assert bool(norm.isfinite().all()) and bool(x.grad.isfinite().all())
    assert bool(weight.grad.isfinite().all())
    torch.cuda.synchronize(device)
    report["gpus"].append({
        "index": device_index,
        "name": torch.cuda.get_device_name(device_index),
        "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        "dense_fa3_forward_backward_vs_math_sdpa": "passed",
        "dense_forward_abs_max": float((out.detach() - ref.detach()).abs().max()),
        "dense_backward_abs_max": max(float((a - t.grad).abs().max()) for a, t in zip(actual_grads, (q, k, v))),
        "varlen_fa3_forward_backward": "passed",
        "fused_triton_rms_forward_backward": "passed",
    })
print(json.dumps(report, indent=2))
