"""per-node health: bf16 matmul TFLOPS on each local GPU + host load. usage: python tests/node_health.py (one process per node, all 8 GPUs)."""
import os
import socket
import time

import torch

out = []
for g in range(torch.cuda.device_count()):
    torch.cuda.set_device(g)
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50):
        a @ b
    torch.cuda.synchronize(); t = (time.time() - t0) / 50
    out.append(2 * 8192 ** 3 / t / 1e12)
la = os.getloadavg()
print(f"[health] {socket.gethostname().split('.')[0]}: TFLOPS per GPU {[round(x) for x in out]} | load {la[0]:.1f} | cpus {os.cpu_count()}", flush=True)
