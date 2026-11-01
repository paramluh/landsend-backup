"""Light-decoder PER-UPDATE overhead (8,1,1 steptime input): one nanostep update's non-compute part = train_ppm._dec_update_finish
(gradient all-reduce of the flat fp32 buffer + clip 0.5 + fused AdamW step) on the FULL-SIZE replicated 0.6B decoder (28 layers), timed
alone, plus its parts (all-reduce only; clip + AdamW only). Synthetic gradients (no data). torchrun-style env (RANK, WORLD_SIZE, ...).
usage: srun ... task_1node.sh <env> test811/bench_dec_update.py [iters=40]"""
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_ppm as T  # noqa: E402
import model_ppm as M  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
dev = torch.device("cuda", T.LOCAL); torch.cuda.set_device(dev)
dist.init_process_group("nccl", device_id=dev)
c4, c06 = M.cfg4b(T.CFG["tower_layers"]), M.cfg06(28)
dec = M.Decoder(c4.hidden_size, c06).to(dev)
for p in dec.parameters():
    assert p.dtype == torch.float32
do = T.DecOpt(dec, "dec_s")
npar = sum(p.numel() for p in dec.parameters())


def timed(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
    for _ in range(N):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / N


def fill():
    do.flat.normal_(0.0, 1e-3)


def full():
    fill(); T._dec_update_finish(do, 0, 8)


def ar_only():
    fill(); dist.all_reduce(do.flat); do.flat.div_(T.WORLD)


def opt_only():
    fill()
    torch.nn.utils.clip_grad_norm_(do.ps, T.CFG["dec_clip"])
    do.opt.step()


t_fill = timed(fill)
r = {"full_update_s": timed(full) - t_fill, "allreduce_s": timed(ar_only) - t_fill, "clip_adamw_s": timed(opt_only) - t_fill}
if T.RANK == 0:
    print(f"[bench] world {T.WORLD}: 0.6B decoder ({npar / 1e9:.3f}B params, flat fp32 {do.flat.numel() * 4 / 1e9:.2f} GB) per-update "
          f"non-compute overhead over {N} iters: " + ", ".join(f"{k} {v * 1000:.1f} ms" for k, v in r.items()), flush=True)
dist.destroy_process_group()
