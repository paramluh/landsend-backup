#!/bin/bash
# 40 s bf16 matmul burn on every visible GPU, sampling SM clocks / temp / throttle reasons every 5 s
nvidia-smi --query-gpu=index,pci.bus_id,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I -c '
import torch, time, threading
n = torch.cuda.device_count()
def burn(i):
    d = torch.device("cuda", i); a = torch.randn(8192, 8192, device=d, dtype=torch.bfloat16); b = torch.randn_like(a)
    t0 = time.time(); k = 0
    while time.time() - t0 < 40:
        c = a @ b; k += 1
        if k % 50 == 0: torch.cuda.synchronize(d)
    torch.cuda.synchronize(d); dt = time.time() - t0
    print(f"gpu{i}: {k * 2 * 8192**3 / dt / 1e12:.0f} TFLOPS bf16", flush=True)
ts = [threading.Thread(target=burn, args=(i,)) for i in range(n)]
[t.start() for t in ts]; [t.join() for t in ts]
' &
P=$!
for i in 1 2 3 4 5 6 7; do sleep 5; nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader | tr '\n' ' '; echo; done
wait $P
