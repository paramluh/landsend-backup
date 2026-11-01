#!/usr/bin/env python3
"""pose6 A1 GPU guard (run before anything is loaded): exit 3 if a throttled GPU is assigned. (1) any assigned PHYSICAL index 6 or 7 on
slinky-4 (thermal fault, memory slinky4-gpus-throttled) per CUDA_VISIBLE_DEVICES / SLURM_JOB_GPUS / scontrol IDX; (2) a 25 s bf16 matmul burn
on every assigned GPU at once, nvidia-smi sampled every 2.5 s: each GPU must reach >= 450 TFLOPS, a median SM clock (last 12 s) >= 50 %
of clocks.max.sm and show no thermal / HW slowdown reason (a full burn is power-capped near 1.4 GHz on a healthy H100: normal)."""
import os
import re
import subprocess
import sys
import threading
import time

import torch

host = os.uname().nodename
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
jg = os.environ.get("SLURM_JOB_GPUS", "") or os.environ.get("SLURM_STEP_GPUS", "")
idx = ""
try:
    sj = subprocess.run(["scontrol", "show", "job", "-d", os.environ.get("SLURM_JOB_ID", "")], capture_output=True, text=True, timeout=20).stdout
    mm = re.search(r"gpu[^ ]*\(IDX:([0-9,\-]+)\)", sj)
    idx = mm.group(1) if mm else ""
except Exception as e:
    print("[guard] scontrol failed:", e)


def ids(s):
    out = set()
    for part in [p for p in s.split(",") if p.strip()]:
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        elif part.strip().isdigit():
            out.add(int(part))
    return out


phys = ids(cvd) | ids(jg) | ids(idx)
print(f"[guard] host {host} CUDA_VISIBLE_DEVICES={cvd!r} SLURM_JOB_GPUS={jg!r} scontrol IDX={idx!r} -> physical {sorted(phys)}", flush=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,clocks_throttle_reasons.active,memory.used",
                      "--format=csv"], capture_output=True, text=True).stdout, flush=True)
if host.startswith("slinky-4") and phys & {6, 7}:
    print(f"[guard] EXIT 3: slinky-4 physical GPU {sorted(phys & {6, 7})} assigned (thermally throttled)", flush=True)
    sys.exit(3)
vis = sorted(ids(cvd))
assert torch.cuda.device_count() == 2 and len(vis) == 2, ("need exactly 2 visible GPUs", torch.cuda.device_count(), cvd)
MATS = []
for i in range(2):                                             # contexts + operands created in the main thread (a lazy context init
    d = torch.device(f"cuda:{i}")                              #   from two threads at once failed: 'CUDA driver error: operation not permitted')
    MATS.append((torch.randn(8192, 8192, device=d, dtype=torch.bfloat16), torch.randn(8192, 8192, device=d, dtype=torch.bfloat16)))
    torch.cuda.synchronize(d)
stop = time.time() + 25.0
flops = [0.0, 0.0]


def burn(i):
    d = torch.device(f"cuda:{i}")
    torch.cuda.set_device(d)
    a, b = MATS[i]
    n = 0
    t0 = time.time()
    while time.time() < stop:
        for _ in range(8):
            a @ b
        torch.cuda.synchronize(d)
        n += 8
    flops[i] = 2 * 8192 ** 3 * n / (time.time() - t0) / 1e12


def burn_safe(i):
    try:
        burn(i)
    except Exception as e:                                     # a burn that cannot run is a failed guard, never a silent pass
        print(f"[guard] burn on cuda:{i} failed: {type(e).__name__}: {e}", flush=True)
        flops[i] = -1.0


th = [threading.Thread(target=burn_safe, args=(i,)) for i in range(2)]
for t in th:
    t.start()
samples = []
while time.time() < stop:
    time.sleep(2.5)
    q = subprocess.run(["nvidia-smi", "-i", ",".join(map(str, vis)), "--query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,clocks_throttle_reasons.active",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
    samples.append((time.time(), [[float(x) if i_ < 4 else int(x.strip(), 16) for i_, x in enumerate(ln.split(","))] for ln in q]))
for t in th:
    t.join()
late = [s for tt, s in samples if tt >= stop - 12.0] or [s for _, s in samples]
bad = []
for g in range(len(vis)):
    clk = sorted(s[g][1] for s in late)
    med, mx, temp = clk[len(clk) // 2], late[-1][g][2], late[-1][g][3]
    therm = any(int(s[g][4]) & 0xE0 for s in late)              # SW thermal slowdown 0x20 / HW slowdown 0x40 / HW power brake 0x80
    print(f"[guard] GPU {int(late[-1][g][0])}: SM clock median {med:.0f} / max {mx:.0f} MHz, {temp:.0f} C, bf16 matmul {flops[g]:.0f} TFLOPS, "
          f"thermal/HW slowdown seen {therm}", flush=True)
    # a full bf16 matmul burn runs power-capped (~1.4 GHz, ~650-700 TFLOPS on a healthy H100: normal); the thermal fault of slinky-4 6-7 sits at
    # 345-420 MHz / ~230 TFLOPS with reason 0x20
    if flops[g] < 450 or med < 0.5 * mx or therm:
        bad.append(int(late[-1][g][0]))
if bad:
    print(f"[guard] EXIT 3: GPU {bad} throttled under load (bf16 matmul < 450 TFLOPS, SM clock < 50 % of max or a thermal / HW slowdown "
          f"reason): latency numbers would be worthless", flush=True)
    sys.exit(3)
print("[guard] OK", flush=True)
