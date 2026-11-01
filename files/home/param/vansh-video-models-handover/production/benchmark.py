"""
Dummy cached production step: Initial + 30 blocks + Final, no audio encoder. Dummy weights, dummy KV cache.
Results and analysis in production/benchmark.md.

  # wan_5b_wide (dim 3840), 120 tokens, full compile sweep (the original table)
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python production/benchmark.py --sweep
  # 2x H and 2x W (480 tokens), d=5120 model, moving frame's k/v projected each step (real cached step)
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python production/benchmark.py --tokens 480 --dim 5120 --kv 1
  # same, tensor parallel over 2 GPUs (WORLD_SIZE > 1 => heads + FFN split, two-shot NVLink all-reduce)
  CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 production/benchmark.py \
      --tokens 480 --dim 5120 --kv 1 2>&1 | grep -v multicast
"""
import argparse
import os
import re
import sys
import traceback

os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F
from flash_attn_interface import flash_attn_func

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=120, choices=(120, 480))   # 30x16 or 60x32 latent, patch 1x2x2
p.add_argument("--dim", type=int, default=3840)
p.add_argument("--ffn", type=int, default=None)                          # default 14/3 * dim, multiple of 256
p.add_argument("--heads", type=int, default=None)                        # default head size 128
p.add_argument("--kv", type=int, default=0)                              # 1: k/v projections of the moving frame
p.add_argument("--comm", default="two_shot", choices=("nccl", "one_shot", "two_shot"))   # TP all-reduce kernel
p.add_argument("--sweep", action="store_true")                           # 1 GPU: reduce-overhead/max-autotune/per-block
p.add_argument("--profile", action="store_true")                         # kernel-time buckets of the graphed step
args = p.parse_args()

DIM, LAYERS = args.dim, 30
FFN = args.ffn or round(DIM * 14 / 3 / 256) * 256
HEADS = args.heads or DIM // 128
HEAD = DIM // HEADS
Q, A = args.tokens, 5
KV = 8 * Q                          # sink + 6 history + self
CACHED = KV - Q if args.kv else KV  # keys already in the cache (all of them when k/v are not projected)
LAT = {120: (30, 16), 480: (60, 32)}[Q]
AUDIO_LAYERS = {4, 7, 10, 13, 16, 19, 22, 25}

rank = int(os.environ.get("RANK", 0))
world = int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dev = torch.device("cuda", torch.cuda.current_device())
TP = world > 1
if TP:
    dist.init_process_group("nccl", device_id=dev)
HL, FL = HEADS // world, FFN // world       # this rank's heads and FFN width
DL = HL * HEAD


def log(msg):
    if rank == 0:
        print(msg, flush=True)


# ----------------------------------------------------------------------------------------------- TP collectives
class Comm:
    """All-reduce of the [1, Q, DIM] bf16 partial residual after o / w2 / audio out_proj. nccl: functional collective.
    one_shot / two_shot: torch symmetric-memory kernels over NVLink; every call site owns one symmetric buffer."""

    def __init__(self):
        self.group = dist.group.WORLD
        self.bufs = []
        if args.comm != "nccl":
            import torch.distributed._symmetric_memory as symm_mem
            self.symm_mem = symm_mem
            symm_mem.enable_symm_mem_for_group(self.group.group_name)

    def site(self, shape, dtype=torch.bfloat16):
        if args.comm == "nccl":
            return None
        buf = self.symm_mem.empty(*shape, dtype=dtype, device=dev)
        self.symm_mem.rendezvous(buf, self.group.group_name)
        self.bufs.append(buf)
        return len(self.bufs) - 1

    def one_shot(self, x, site):
        buf = self.bufs[site]
        buf.copy_(x)
        return torch.ops.symm_mem.one_shot_all_reduce(buf, "sum", self.group.group_name)

    def all_reduce(self, x, site):
        if args.comm == "nccl":
            return funcol.all_reduce(x, "sum", self.group)
        if args.comm == "two_shot":
            return two_shot_all_reduce(x, site)
        return self.one_shot(x, site)

    def all_reduce_f32(self, x, site):
        """A few KB of fp32 (qk-norm sums of squares): latency bound, one-shot."""
        if args.comm == "nccl":
            return funcol.all_reduce(x, "sum", self.group)
        return self.one_shot(x, site)


comm = Comm() if TP else None


# two_shot_all_reduce_ is in-place and Dynamo cannot trace it (graph break per call site). Wrapped as an opaque op
# that returns the site's own symmetric buffer, so the whole step still compiles into one graph.
@torch.library.custom_op("benchmark::two_shot_all_reduce", mutates_args=())
def two_shot_all_reduce(x: torch.Tensor, site: int) -> torch.Tensor:
    buf = comm.bufs[site]
    buf.copy_(x)
    torch.ops.symm_mem.two_shot_all_reduce_(buf, "sum", comm.group.group_name)
    return buf


@two_shot_all_reduce.register_fake
def _(x, site):
    return torch.empty_like(x)


# ----------------------------------------------------------------------------------------------- model
class RMSNorm(nn.Module):
    """RMS over all heads jointly (as in the real model). Under TP the sum of squares is all-reduced: exact."""

    def __init__(self, local_dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(local_dim))

    def forward(self, x, ss):
        return x * torch.rsqrt(ss / DIM + self.eps).to(x.dtype) * self.weight


class AudioBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_x = nn.LayerNorm(DIM, elementwise_affine=False)
        self.norm_a = nn.LayerNorm(DIM, elementwise_affine=False)
        self.q_proj = nn.Linear(DIM, DL)
        self.kv_proj = nn.Linear(DIM, DL * 2)
        self.norm_q = RMSNorm(DL)
        self.norm_k = RMSNorm(DL)
        self.out_proj = nn.Linear(DL, DIM, bias=rank == 0)     # reduced projections: bias on one rank only
        if TP:
            self.site = comm.site((1, Q, DIM))
            self.site_ss = comm.site((1, Q + A), torch.float32)

    def forward(self, x, audio):
        q = self.q_proj(self.norm_x(x))
        k, v = self.kv_proj(self.norm_a(audio)).chunk(2, dim=-1)
        ss = torch.cat((q.float().pow(2).sum(-1), k.float().pow(2).sum(-1)), dim=1)
        if TP:
            ss = comm.all_reduce_f32(ss, self.site_ss)
        q = self.norm_q(q, ss[:, :Q, None]).view(1, Q, HL, HEAD)
        k = self.norm_k(k, ss[:, Q:, None]).view(1, A, HL, HEAD)
        out = self.out_proj(flash_attn_func(q, k, v.view(1, A, HL, HEAD)).flatten(2))
        return comm.all_reduce(out, self.site) if TP else out


class Block(nn.Module):
    def __init__(self, audio):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.q = nn.Linear(DIM, DL)
        if args.kv:
            self.k = nn.Linear(DIM, DL)
            self.v = nn.Linear(DIM, DL)
        self.o = nn.Linear(DL, DIM, bias=rank == 0)
        self.w1 = nn.Linear(DIM, FL)
        self.w2 = nn.Linear(FL, DIM, bias=rank == 0)
        self.modulation = nn.Parameter(torch.randn(1, 6, DIM) / DIM ** 0.5)
        self.audio = audio
        if TP:
            self.site_attn = comm.site((1, Q, DIM))
            self.site_ffn = comm.site((1, Q, DIM))

    def forward(self, x, e0, k, v, audio):
        m = self.modulation + e0
        h = self.norm1(x) * (1 + m[:, 1]) + m[:, 0]
        q = self.q(h).view(1, Q, HL, HEAD)
        if args.kv:
            k = torch.cat((k, self.k(h).view(1, Q, HL, HEAD)), dim=1)
            v = torch.cat((v, self.v(h).view(1, Q, HL, HEAD)), dim=1)
        a = self.o(flash_attn_func(q, k, v).flatten(2))
        if TP:
            a = comm.all_reduce(a, self.site_attn)
        x = x + a * m[:, 2]
        if self.audio is not None:
            x = x + self.audio(x, audio)
        h = self.norm2(x) * (1 + m[:, 4]) + m[:, 3]
        f = self.w2(F.gelu(self.w1(h), approximate="tanh"))
        if TP:
            f = comm.all_reduce(f, self.site_ffn)
        return x + f * m[:, 5]


class Step(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv3d(48, DIM, (1, 2, 2), (1, 2, 2))
        self.pose = nn.Conv3d(48, DIM, (1, 2, 2), (1, 2, 2))
        self.time = nn.Sequential(nn.Linear(256, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(DIM, DIM * 6))
        self.blocks = nn.ModuleList(Block(AudioBlock() if i in AUDIO_LAYERS else None) for i in range(LAYERS))
        self.head_norm = nn.LayerNorm(DIM, elementwise_affine=False)
        self.head = nn.Linear(DIM, 192)
        self.head_mod = nn.Parameter(torch.randn(1, 2, DIM) / DIM ** 0.5)

    def forward(self, latent, pose, t_emb, k, v, audio):
        x = self.patch(latent).flatten(2).transpose(1, 2) + self.pose(pose).flatten(2).transpose(1, 2)
        e = self.time(t_emb)
        e0 = self.time_proj(e).unflatten(-1, (6, DIM))
        for block in self.blocks:
            x = block(x, e0, k, v, audio)
        m = self.head_mod + e.unsqueeze(1)
        return self.head(self.head_norm(x) * (1 + m[:, 1]) + m[:, 0])


def make_step():
    torch.manual_seed(0)
    return Step().to(dev).bfloat16().eval()


def make_args():
    return (
        torch.randn(1, 48, 1, *LAT, device=dev, dtype=torch.bfloat16),
        torch.randn(1, 48, 1, *LAT, device=dev, dtype=torch.bfloat16),
        torch.randn(1, 256, device=dev, dtype=torch.bfloat16),
        torch.randn(1, CACHED, HL, HEAD, device=dev, dtype=torch.bfloat16),
        torch.randn(1, CACHED, HL, HEAD, device=dev, dtype=torch.bfloat16),
        torch.randn(1, A, DIM, device=dev, dtype=torch.bfloat16),
    )


def analytical_tflop():
    """GEMM FLOPs on the Q tokens (audio k/v on A tokens and time MLP are negligible) + attention FLOPs."""
    per_token = (LAYERS * ((4 if args.kv else 2) * DIM * DIM + 2 * DIM * FFN) + len(AUDIO_LAYERS) * 2 * DIM * DIM
                 + 2 * 48 * 4 * DIM + DIM * 192)
    attn = LAYERS * 4 * Q * KV * DIM + len(AUDIO_LAYERS) * 4 * Q * A * DIM
    return (2 * Q * per_token + attn) / 1e12


# ----------------------------------------------------------------------------------------------- timing
def time_ms(fn, n=40, warmup=15):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        if TP:
            dist.barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def cuda_graph(fn, warmup=5):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()

        def replay():
            g.replay()

        return replay


def bench(name, fn, **kw):
    try:
        ms = time_ms(fn, **kw)
        log(f"{name:42s} {ms:7.3f} ms/step")
        return ms
    except Exception as exc:
        log(f"{name:42s} FAIL {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


BUCKETS = (("attention", r"flash|fmha"), ("collective", r"nccl|all_reduce|one_shot|two_shot|multimem|symm"),
           ("gemm", r"nvjet|cutlass|gemm|xmma|cublas"), ("copy", r"memcpy|copy_"))


def profile(fn, n=20):
    """Device kernel time of one step by category (the residual bucket is fused elementwise: AdaLN, LN, GELU, ...)."""
    from torch.profiler import ProfilerActivity, profile as torch_profile
    with torch.inference_mode(), torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    buckets = {}
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        cat = next((c for c, pat in BUCKETS if re.search(pat, ev.name, re.I)), "elementwise")
        buckets[cat] = buckets.get(cat, 0.0) + ev.time_range.elapsed_us() / n / 1e3
    for cat, ms in sorted(buckets.items(), key=lambda kv: -kv[1]):
        log(f"  {cat:12s} {ms:7.3f} ms/step")
    log(f"  {'kernel sum':12s} {sum(buckets.values()):7.3f} ms/step")


def compile_blocks(step, **kwargs):
    for block in step.blocks:
        block.forward = torch.compile(block.forward, dynamic=False, **kwargs)
    return step


def main():
    assert not (args.sweep and TP), "--sweep is single GPU"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)
    tflop = analytical_tflop()
    results = {}

    step = make_step()
    n_params = sum(p.numel() for p in step.parameters()) / 1e9
    log(f"dim {DIM} ffn {FFN} heads {HEADS}x{HEAD} layers {LAYERS} | Q {Q} (latent {LAT[0]}x{LAT[1]}) KV {KV} "
        f"(cached {CACHED}) kv_proj {args.kv} | {n_params:.2f}B params"
        f"{f'/GPU, TP over {world} GPUs, all-reduce {args.comm}' if TP else ''} | analytical {tflop:.2f} TFLOP/step")
    a = make_args()
    run = lambda: step(*a)
    results["eager"] = bench("eager", run)
    results["eager+graph"] = bench("eager+cudagraph", cuda_graph(run))
    del step
    torch.cuda.empty_cache()

    step = make_step()
    compiled = torch.compile(step, dynamic=False)
    a = make_args()
    run = lambda: compiled(*a)
    results["whole/default"] = bench("whole default", run, warmup=20)
    g = cuda_graph(run, warmup=20)
    results["whole/default+graph"] = bench("whole default+cudagraph", g)
    if args.profile:
        profile(g)
    del step, compiled, g
    torch.cuda.empty_cache()

    if args.sweep:
        for mode, extra_warmup in (("reduce-overhead", 30), ("max-autotune", 50)):
            step = make_step()
            log(f"--- compile whole-step {mode} ---")
            compiled = torch.compile(step, dynamic=False, mode=mode)
            a = make_args()
            run = lambda: compiled(*a)
            results[f"whole/{mode}"] = bench(f"whole {mode}", run, warmup=extra_warmup)
            del step, compiled
            torch.cuda.empty_cache()
        for mode, extra_warmup in ((None, 25), ("max-autotune-no-cudagraphs", 50)):
            step = make_step()
            label = mode or "default"
            log(f"--- compile per-block {label} ---")
            compile_blocks(step, **({"mode": mode} if mode else {}))
            a = make_args()
            run = lambda: step(*a)
            results[f"blocks/{label}"] = bench(f"blocks {label}", run, warmup=extra_warmup)
            try:
                results[f"blocks/{label}+graph"] = bench(f"blocks {label}+cudagraph", cuda_graph(run, warmup=extra_warmup))
            except Exception as exc:
                log(f"blocks {label}+cudagraph FAIL {type(exc).__name__}: {exc}")
            del step
            torch.cuda.empty_cache()

    ok = {k: v for k, v in results.items() if v is not None}
    best = min(ok, key=ok.get)
    log(f"\nFASTEST: {best}  {ok[best]:.3f} ms/step  ->  {tflop / ok[best] * 1e3:.0f} TFLOP/s"
        f"{f' aggregate over {world} GPUs' if TP else ''}; peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        f"{'/GPU' if TP else ''}")
    if TP:
        dist.barrier()
        torch.cuda.synchronize()
        os._exit(0)     # destroy_process_group() hangs on this NCCL build


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
