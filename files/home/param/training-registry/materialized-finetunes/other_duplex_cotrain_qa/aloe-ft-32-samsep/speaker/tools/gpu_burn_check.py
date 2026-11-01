"""host pick (10-24): ~12 s bf16 matmul burst on every visible GPU at once, SM clock + temperature sampled under load (nvidia-smi), TFLOPS
per GPU. A throttled GPU (cf. slinky-4 GPUs 6-7 at 345 MHz under load) shows a low clock / low TFLOPS. usage: python -I gpu_burn_check.py"""
import socket
import subprocess
import threading
import time

import torch

H = socket.gethostname().split(".")[0]
n = torch.cuda.device_count()
res = {}


def burn(i):
    d = torch.device("cuda", i)
    a = torch.randn(8192, 8192, device=d, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=d, dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(d)
    t0, k = time.time(), 0
    while time.time() - t0 < 12.0:
        for _ in range(20):
            a @ b
        torch.cuda.synchronize(d)
        k += 20
    res[i] = 2 * 8192 ** 3 * k / (time.time() - t0) / 1e12


th = [threading.Thread(target=burn, args=(i,)) for i in range(n)]
for t in th:
    t.start()
time.sleep(8.0)
q = subprocess.run(["nvidia-smi", "--query-gpu=index,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active",
                    "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().split("\n")
for t in th:
    t.join()
print(H, " | ".join(f"gpu{i} {res[i]:.0f}TF [{q[i].split(', ', 1)[1]}]" for i in range(n)), flush=True)
