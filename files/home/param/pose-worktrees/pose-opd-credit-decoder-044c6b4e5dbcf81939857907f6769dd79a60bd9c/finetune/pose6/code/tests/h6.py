"""pose6 B1 tests: shared CPU harness (TEST ONLY; never imported by the trainer). = pose5/code-811-s2p/tests/s2p_postmerge/s2p_cpu.py's
reference kernels + digests (the trainer is GPU-only: fused Triton rms / add_rms / qk_prep / silu_mul, FA3 varlen, bf16 autocast 'cuda',
pinned memory). EVERY variant of every comparison runs the SAME replacements, so comparisons are unaffected; the production GPU maths is not
exercised here (and is unchanged by the port when the flags are off: same op sequence, proven on CPU)."""
import hashlib
import os
import socket
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

P6 = "/data/home/amartyavarma/ppm/pose6/code"
MONO = "/data/home/amartyavarma/ppm/aloe-pose-mono-8020/code"
C811 = "/data/home/amartyavarma/ppm/pose5/code-811-s2p"
SIDECAR = "/data/home/amartyavarma/ppm/pose6/g0aloe/g0aloe.parquet"
OUTD = f"{P6}/tests/out"
SCR = os.environ.get("P6_SCR", "/scratch/amartyavarma/pose6-b1-tests")
NEW = ("inp.g0.weight", "tower_p.s_emb.weight")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s {socket.gethostname().split('.')[0]}]", *a, flush=True)


def use_code(code):
    for p in (f"{code}/vendor", f"{code}/lib", code):
        sys.path.insert(0, p)


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


def addmm_cpu(inp, a, b_, *args, out_dtype=None, out=None, **kw):
    if out_dtype is None:
        return _addmm(inp, a, b_, *args, out=out, **kw) if out is not None else _addmm(inp, a, b_, *args, **kw)
    r = _addmm(inp.to(out_dtype), a.to(out_dtype), b_.to(out_dtype), *args, **kw)
    return out.copy_(r) if out is not None else r


def mm_cpu(a, b_, *args, out_dtype=None, **kw):
    return _mm(a, b_, *args, **kw) if out_dtype is None else _mm(a.to(out_dtype), b_.to(out_dtype), *args, **kw)


class AutocastCPU(_AC):
    def __init__(self, device_type, *a, **k):
        super().__init__("cpu" if device_type == "cuda" else device_type, *a, **k)


def cpu_patches():
    """install the replacements (after model_ppm / trainer import; the vendored modules are shared, byte-identical files)."""
    FO, fa3_op = sys.modules["fused_triton"], sys.modules["fa3_op"]
    torch.addmm, torch.mm = addmm_cpu, mm_cpu
    FO.rms, FO.add_rms, FO.qk_prep, FO.silu_mul = rms, add_rms, qk_prep, silu_mul
    fa3_op.fa3_varlen_func = fa3_varlen
    torch.autocast = AutocastCPU
    torch.Tensor.pin_memory = lambda self, *a, **k: self


def init_dist(port, rank=0, world=1):
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{int(port)}", rank=rank, world_size=world)
    return dist


# ------------------------------------------------------------------------------------------------ digests
def _flat(t):
    t = t.detach()
    t = t.to_local() if hasattr(t, "to_local") else t
    return t.contiguous().cpu().reshape(-1)


def sha(t):
    if t is None:
        return None
    return hashlib.sha256(_flat(t).view(torch.uint8).numpy().tobytes()).hexdigest()[:32]


def loc(t):
    return t.to_local() if hasattr(t, "to_local") else t


# ------------------------------------------------------------------------------------------------ tiny configs + seeded init
def tiny_cfgs(M):
    """811 shapes reduced (= s2p_cpu.tiny_cfgs): backbone 2 x H128, towers 2 x H64 (head_dim 32 = the backbone's: the Aloe trainer passes
    the backbone's Rot to the towers), decoders 1 x H64; real vocab 151,936 / tower vocab 151,682."""
    c32 = M.cfg32(2)
    c32.hidden_size, c32.intermediate_size, c32.num_attention_heads, c32.num_key_value_heads, c32.head_dim = 128, 256, 4, 2, 32
    c4 = M.cfg4b(2)
    c4.hidden_size, c4.intermediate_size, c4.num_attention_heads, c4.num_key_value_heads, c4.head_dim = 64, 128, 4, 2, 32
    c06 = M.cfg06(1)
    c06.hidden_size, c06.intermediate_size, c06.num_attention_heads, c06.num_key_value_heads, c06.head_dim = 64, 128, 4, 2, 16
    return c32, c4, c06


def seeded_init(M, model, c4, c06, new="zero", seed=20261027):
    """every tensor its init rule (model_ppm.init_rule; Qwen3-32B tensors N(0, 0.02) / ones), seeded per NAME + global row chunk
    (H2.init_rows_: world-invariant): every code variant gets bit-identical shared tensors. new: 'zero' (the init rule of the mods) | 'nz'
    (s_emb N(0, 0.02 / sqrt 32), g0 N(0, 0.02))."""
    H2 = M.H2
    with torch.no_grad():
        for n, p in model.named_parameters():
            n_ = n.replace("_checkpoint_wrapped_module.", "")
            shp = tuple(p.shape)
            if M.hf_key(n_) is not None:
                kind, std = ("ones", 0.0) if n_.endswith("norm.weight") else ("normal", 0.02)
            else:
                kind, std = M.init_rule(n_, shp, c4, c06, 0.02)
            if n_ in NEW:
                assert (kind, std) == ("zeros", 0.0), ("init_rule must zero-init the mods' params", n_, kind, std)
                if new == "nz":
                    kind, std = "normal", (0.02 / 32 ** 0.5 if n_ == "tower_p.s_emb.weight" else 0.02)
            a, _ = M._rows(p)
            H2.init_rows_(loc(p).data, a, shp, kind, std, n_, seed)


def full_seeded(M, name, shape, kind, std, seed, dtype=torch.float32):
    """the full tensor init_rows_ produces for (name, shape) on any world (row0 = 0, all rows)."""
    t = torch.empty(shape, dtype=dtype)
    M.H2.init_rows_(t, 0, tuple(shape), kind, std, name, seed)
    return t


def counts(b):
    """(text targets, cb0 labels, pose cb0 labels) of a micro (for prep's normalisers)."""
    return float(len(b["sel"])), float((np.asarray(b["cb0"]) >= 0).sum()), float((np.asarray(b["pcb0"]) >= 0).sum()) if "pcb0" in b else 1.0
