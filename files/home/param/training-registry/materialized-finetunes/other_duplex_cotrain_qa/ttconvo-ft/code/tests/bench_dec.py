"""T14 decoder-update microbenchmark: K nanosteps of ONE full-size 0.6B decoder (model_ppm.Decoder, 28 layers) on synthetic tower outputs
(hv [N, 2560], Mimi-like codes), FSDP2 variants, timed per nanostep (after warmup). torchrun / srun over 1..n nodes.
  V0 current: shard over the WORLD, per-layer units, reshard after forward, activation checkpointing, eager
  V1 + no reshard after forward (one all-gather per nanostep instead of forward + backward re-gather)
  V2 + no activation checkpointing in the decoder
  V3 + HSDP: shard within the node, replicate across nodes (all-gather / reduce-scatter on NVLink, cross-node all-reduce of the shard grads)
  V4 + compiled layers + static padded chunks (lightfast_s3q style)
usage: python tests/bench_dec.py [variants=V0,V1,V2,V3,V4] [N=7500] [K=25]"""
import os
import sys
import time
import types

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_ppm as M  # noqa: E402
import lightms_s3q as LMS  # noqa: E402

VAR = (sys.argv[1] if len(sys.argv) > 1 else "V0,V1,V2,V3,V4").split(",")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 7500
K = int(sys.argv[3]) if len(sys.argv) > 3 else 25
RANK, WORLD, LOCAL = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
LW = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))


def build(v, c06, dev, mesh_w, mesh_h):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    mpn = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    with torch.device("meta"):
        d = M.Decoder(2560, c06)
    lay = d.model.layers
    if v in ("V0", "V1"):
        for i in range(len(lay)):
            lay[i] = checkpoint_wrapper(lay[i], checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False)
    mesh = mesh_h if v in ("V3", "V4") else mesh_w
    raf = v == "V0"
    for L in lay:
        fully_shard(L, mesh=mesh, mp_policy=mp, reshard_after_forward=raf)
    fully_shard(d, mesh=mesh, mp_policy=mpn, reshard_after_forward=raf)
    d.to_empty(device=dev)
    d.model.rotary_emb = type(d.model.rotary_emb)(config=c06, device=dev)
    for n, p in d.named_parameters():
        kind, std = M._light_rule("light." + n.replace("_checkpoint_wrapped_module.", ""), tuple(p.shape), types.SimpleNamespace(hidden_size=2560), c06)
        a, _ = M._rows(p) if p.placements[-1].is_shard() else (0, 0)
        M.H2.init_rows_(M._loc(p), a, tuple(p.shape), kind, std, n, 1)
    if v == "V4":
        for L in lay:
            L.forward = torch.compile(L.forward, mode=os.environ.get("BENCH_MODE", "default"), fullgraph=False, dynamic=False)
    return d


class Rep:
    """replicated decoder: fp32 masters on every GPU, bf16 autocast compute, ONE flat fp32 gradient all-reduce (mean) per update."""

    def __init__(self, d, mode):
        self.d = d
        self.ps = list(d.parameters())
        n = sum(p.numel() for p in self.ps)
        self.flat = torch.zeros(n, dtype=torch.float32, device=self.ps[0].device)
        o = 0
        for p in self.ps:
            p.grad = self.flat[o:o + p.numel()].view_as(p); o += p.numel()
        f = lambda *a: M.Decoder.forward(d, *a)  # noqa: E731
        self.f = torch.compile(f, mode=mode, fullgraph=False, dynamic=False) if mode else f

    def __call__(self, *a):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.f(*a)

    def parameters(self):
        return self.ps


def run_rep(r, hv, depth, light, reps=2):
    opt = torch.optim.AdamW([{"params": r.ps, "lr": 1e-4}], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
    n = light.numel(); ch = -(-n // K); ts = []
    W = dist.get_world_size()
    for rr in range(reps + 1):
        hd = hv.detach().requires_grad_()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(rr)).to(hv.device)
        torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
        for j in range(K):
            c = perm[j::K]; m = c.numel()
            fl = light.index_select(0, c)
            if m < ch:
                fl = torch.cat([fl, light[:1].expand(ch - m)])
            wf = torch.zeros(ch, device=hv.device); wf[:m] = 1.0 / (31.0 * m)
            r.flat.zero_(); hd.grad = None
            loss = r(hd, depth, fl, wf, m)[0]
            loss.backward()
            dist.all_reduce(r.flat); r.flat.div_(W)
            torch.nn.utils.clip_grad_norm_(r.ps, 0.5, foreach=True)
            opt.step()
        torch.cuda.synchronize(); t = time.time() - t0
        if rr:
            ts.append(t)
    t = torch.tensor([sum(ts) / len(ts)], device=hv.device); dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t) / K


def run(d, hv, depth, light, static, reps=2):
    ps = list(d.parameters())
    opt = torch.optim.AdamW([{"params": ps, "lr": 1e-4}], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
    n = light.numel()
    ch = -(-n // K)
    ts = []
    for r in range(reps + 1):
        hd = hv.detach().requires_grad_()

        def ll(c):
            fl = light.index_select(0, c.to(light.device))
            m = fl.numel()
            if static and m < ch:
                fl = torch.cat([fl, light[:1].expand(ch - m)])
            wf = torch.zeros(fl.numel(), device=hv.device)
            wf[:m] = 1.0 / (31.0 * m)
            return d(hd, depth, fl, wf)[0], None
        torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
        LMS.micro_steps(ll, hd, n, K, torch.Generator().manual_seed(r), opt, ps, 0.5, lambda s: 1e-4, r)
        torch.cuda.synchronize(); t = time.time() - t0
        if r:
            ts.append(t)
    t = torch.tensor([sum(ts) / len(ts)], device=hv.device); dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t) / K


def main():
    dev = torch.device("cuda", LOCAL); torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    from torch.distributed.device_mesh import init_device_mesh
    mesh_w = init_device_mesh("cuda", (WORLD,))
    mesh_h = init_device_mesh("cuda", (WORLD // LW, LW), mesh_dim_names=("rep", "shard"))
    c06 = M.cfg06()
    g = torch.Generator(device="cpu").manual_seed(RANK)
    hv = (torch.randn(N, 2560, generator=g)).to(dev, torch.bfloat16)
    depth = (torch.randint(0, 2048, (N, 32), generator=g) + torch.arange(32) * 2048).to(dev)
    light = torch.arange(N, device=dev)
    for v in VAR:
        if v in ("V5", "V6", "V7"):                                  # replicated: V5 eager, V6 compiled default, V7 compiled reduce-overhead (CUDA graphs)
            d = M.Decoder(2560, c06).to(dev)
            torch.manual_seed(0)
            for p in d.parameters():
                dist.broadcast(p.data, 0)
            t = run_rep(Rep(d, {"V5": None, "V6": "default", "V7": "reduce-overhead"}[v]), hv, depth, light)
        else:
            d = build(v, c06, dev, mesh_w, mesh_h)
            t = run(d, hv, depth, light, static=(v == "V4"))
        if RANK == 0:
            print(f"[bench] world {WORLD}: {v}: {1000 * t:.1f} ms / nanostep -> {75 * t:.1f} s per step (75 nanosteps); "
                  f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
        del d
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
