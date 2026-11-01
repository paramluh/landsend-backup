"""pose5 s2p tests: shared CPU harness (TEST ONLY; never imported by the trainer).

The 8,1,1 trainer is GPU-only (vendor fused_triton rms / add_rms / qk_prep / silu_mul, pinned FA3 varlen, bf16 autocast on 'cuda', pinned
host memory). These tests run on spare-node CPUs, so the GPU kernels are replaced IN THE TEST PROCESS ONLY by torch reference
implementations (= ppm/pose-s2p/tests/test_train_path.py's), torch.autocast('cuda') -> torch.autocast('cpu') (same bf16 policy role), and
Tensor.pin_memory -> identity. EVERY variant of every comparison (original code-811 and the s2p copy) runs the SAME replacements, so the
comparisons are unaffected; the production maths (GPU) is not exercised here and is unchanged by the s2p port when the flag is off.

Paths: ORIG = the untouched ppm/code-811 (read-only), COPY = pose5/code-811-s2p. SCR = node-local scratch of the test node (micros, reduced
real weights, synthetic checkpoints: never /data); MICD = the built real micros (data, /data)."""
import hashlib
import os
import socket
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ORIG = "/data/home/amartyavarma/ppm/code-811"
COPY = "/data/home/amartyavarma/ppm/pose5/code-811-s2p"
SCR = "/scratch/amartyavarma/pose5-s2p-test"
MICD = "/data/home/amartyavarma/ppm/pose5/s2p-test-micros"          # the real micros (data pickles, shared by every test node)
OUTD = f"{COPY}/tests/s2p/out"
S2P_KEY = "tower_p.s_emb.weight"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s {socket.gethostname().split('.')[0]}]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ CPU reference kernels (test only)
def rms(x, w, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w.to(x.dtype) * y.to(x.dtype)


def add_rms(h, a, w, eps):
    s = h + a
    return s, rms(s, w, eps)


def qk_prep(q, k, qw, kw, eps, cos, sin):
    hd = qw.shape[0]
    q = q.view(*q.shape[:-1], -1, hd)
    k = k.view(*k.shape[:-1], -1, hd)
    q, k = rms(q, qw, eps), rms(k, kw, eps)
    c, s = cos.unsqueeze(2).to(q.dtype), sin.unsqueeze(2).to(q.dtype)

    def rot(x):
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        return torch.cat([-x2, x1], -1)
    return q * c + rot(q) * s, k * c + rot(k) * s


def silu_mul(g, u):
    return F.silu(g) * u


def fa3_varlen(q, k, v, cuq, cuk, mq, mk, scale, causal):
    out = []
    cu = cuq.tolist()
    rep = q.shape[1] // k.shape[1]
    for a, b in zip(cu[:-1], cu[1:]):
        qq = q[a:b].transpose(0, 1)[None]
        kk = k[a:b].repeat_interleave(rep, 1).transpose(0, 1)[None]
        vv = v[a:b].repeat_interleave(rep, 1).transpose(0, 1)[None]
        o = F.scaled_dot_product_attention(qq, kk, vv, is_causal=bool(causal), scale=scale)
        out.append(o[0].transpose(0, 1))
    return torch.cat(out, 0)


_addmm, _mm, _AC = torch.addmm, torch.mm, torch.autocast


def addmm_cpu(inp, a, b_, *args, out_dtype=None, out=None, **kw):     # heads_s2q CE backward: bf16 GEMM, fp32 accumulate (CUDA-only op)
    if out_dtype is None:
        return _addmm(inp, a, b_, *args, out=out, **kw) if out is not None else _addmm(inp, a, b_, *args, **kw)
    r = _addmm(inp.to(out_dtype), a.to(out_dtype), b_.to(out_dtype), *args, **kw)
    return out.copy_(r) if out is not None else r


def mm_cpu(a, b_, *args, out_dtype=None, **kw):
    return _mm(a, b_, *args, **kw) if out_dtype is None else _mm(a.to(out_dtype), b_.to(out_dtype), *args, **kw)


class AutocastCPU(_AC):
    """torch.autocast('cuda', bf16) of the decoders -> the CPU autocast with the same dtype (the decoders' fp32 masters compute in bf16)."""

    def __init__(self, device_type, *a, **k):
        super().__init__("cpu" if device_type == "cuda" else device_type, *a, **k)


def cpu_patches():
    """install the replacements (call AFTER the trainer / model_ppm import; vendor modules are shared, identical files)."""
    FO, fa3_op = sys.modules["fused_triton"], sys.modules["fa3_op"]
    torch.addmm, torch.mm = addmm_cpu, mm_cpu
    FO.rms, FO.add_rms, FO.qk_prep, FO.silu_mul = rms, add_rms, qk_prep, silu_mul
    fa3_op.fa3_varlen_func = fa3_varlen
    torch.autocast = AutocastCPU
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    if os.environ.get("S2P_TEST_CE_CHUNK"):                      # memory knob for tiny CPU slots: rows per lm_head CE chunk (heads_s2q.CHUNK,
        sys.modules["heads_s2q"].CHUNK = int(os.environ["S2P_TEST_CE_CHUNK"])   # production 1024); every compared variant uses the same value


class _CudaProxy:
    def __init__(self, c):
        self._c = c

    def __getattr__(self, n):
        return getattr(self._c, n)

    def current_device(self):
        return 0


class TorchProxy:
    """safety_ppm's module-level `torch` in the resume tests: tensors it puts on 'cuda:<dev>' go to the CPU (gloo collectives)."""

    def __init__(self, t):
        self._t, self.cuda = t, _CudaProxy(t.cuda)

    def __getattr__(self, n):
        return getattr(self._t, n)

    def tensor(self, *a, device=None, **k):
        return self._t.tensor(*a, device="cpu" if device is not None and str(device).startswith("cuda") else device, **k)


def init_dist(port):
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{int(port)}", rank=0, world_size=1)
    return dist


# ------------------------------------------------------------------------------------------------ digests
def _flat(t):
    t = t.detach()
    t = t.to_local() if hasattr(t, "to_local") else t
    return t.contiguous().cpu().reshape(-1)


CH_ = 1 << 24                                                    # elements per hashed chunk (streamed: same digest as one-shot, low memory)


def sha(t):
    """sha256 of the tensor's bytes (streamed over chunks: identical to hashing all bytes at once)."""
    if t is None:
        return None
    f, h = _flat(t), hashlib.sha256()
    for i in range(0, f.numel(), CH_):
        h.update(f[i:i + CH_].view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:32]


def vsha(t):
    """value-level digest (fp64 copy + 0.0: -0.0 == +0.0), streamed over chunks."""
    if t is None:
        return None
    f, h = _flat(t), hashlib.sha256()
    for i in range(0, f.numel(), CH_):
        h.update((f[i:i + CH_].double() + 0.0).numpy().tobytes())
    return h.hexdigest()[:32]


def loc(t):
    return t.to_local() if hasattr(t, "to_local") else t


# ------------------------------------------------------------------------------------------------ tiny configs (811 shapes reduced)
def tiny_cfgs(M, mini=False):
    """mini (the resume test: 3 GB CPU slots hold the model + AdamW + a loaded checkpoint): backbone H64, towers H32."""
    c32 = M.cfg32(2)
    c32.hidden_size, c32.intermediate_size, c32.num_attention_heads, c32.num_key_value_heads, c32.head_dim = (64, 128, 2, 1, 32) if mini else (128, 256, 4, 2, 32)
    c4 = M.cfg4b(2)                                                # same head_dim as c32: run_micro passes the backbone's Rot to the towers
    c4.hidden_size, c4.intermediate_size, c4.num_attention_heads, c4.num_key_value_heads, c4.head_dim = (32, 64, 1, 1, 32) if mini else (64, 128, 4, 2, 32)
    c06 = M.cfg06(1)
    c06.hidden_size, c06.intermediate_size, c06.num_attention_heads, c06.num_key_value_heads, c06.head_dim = 64, 128, 4, 2, 16
    return c32, c4, c06


def seeded_init(M, model, c4, c06, s_emb="zero", seed=20261027):
    """every tensor its 811 init rule (model_ppm.init_rule; the Qwen3-32B tensors N(0, 0.02) / ones instead of the safetensors), seeded per
    NAME (H2.init_rows_): the original and the s2p copy get bit-identical shared tensors. s_emb: 'zero' (= the copy's init_rule) | 'nz'
    (N(0, 0.02 / sqrt(32)), the code_emb rule) for the non-zero check."""
    H2 = M.H2
    with torch.no_grad():
        for n, p in model.named_parameters():
            n_ = n.replace("_checkpoint_wrapped_module.", "")
            shp = tuple(p.shape)
            if M.hf_key(n_) is not None:
                kind, std = ("ones", 0.0) if n_.endswith("norm.weight") else ("normal", 0.02)
            else:
                kind, std = M.init_rule(n_, shp, c4, c06, 0.02)
            if n_ == S2P_KEY:
                assert (kind, std) == ("zeros", 0.0), ("the copy's init_rule must zero-init s_emb", kind, std)
                if s_emb == "nz":
                    kind, std = "normal", 0.02 / 32 ** 0.5
            a, _ = M._rows(p)
            H2.init_rows_(loc(p).data, a, shp, kind, std, n_, seed)
