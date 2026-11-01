"""NCCL bandwidth check for FSDP-sized collectives (bf16 all-gather, fp32 reduce-scatter) over the job's ranks. usage: srun ... task_ppm.sh bench_nccl.py"""
import os
import time

import torch
import torch.distributed as dist

RANK, WORLD, LOCAL = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
dev = torch.device("cuda", LOCAL); torch.cuda.set_device(dev)
dist.init_process_group("nccl", device_id=dev)
for nbytes, dt, op in ((488e6 * 2, torch.bfloat16, "ag"), (488e6 * 4, torch.float32, "rs"), (575e6 * 4, torch.float32, "ar")):   # 32B layer; 0.6B decoder grads
    n = int(nbytes // torch.tensor([], dtype=dt).element_size()) // WORLD * WORLD
    full = torch.empty(n, dtype=dt, device=dev); part = torch.empty(n // WORLD, dtype=dt, device=dev)
    for i in range(8):
        if i == 3:
            torch.cuda.synchronize(); t0 = time.time()
        if op == "ag":
            dist.all_gather_into_tensor(full, part)
        elif op == "rs":
            dist.reduce_scatter_tensor(part, full)
        else:
            dist.all_reduce(full)
    torch.cuda.synchronize(); t = (time.time() - t0) / 5
    bus = nbytes * (WORLD - 1) / WORLD / t / 1e9 * (2 if op == "ar" else 1)
    if RANK == 0:
        print(f"[nccl] world {WORLD} hosts {os.environ.get('PPM_HOSTS')}: {op} {nbytes / 1e9:.2f} GB {dt}: {1000 * t:.1f} ms, bus bw {bus:.0f} GB/s", flush=True)
dist.destroy_process_group()
