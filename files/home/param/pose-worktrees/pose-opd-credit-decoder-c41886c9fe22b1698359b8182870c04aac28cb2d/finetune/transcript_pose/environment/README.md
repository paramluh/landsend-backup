# Training environment

Python 3.12.3 at `/home/param/pose-repo/.venv` on all seven landsend nodes.
`requirements.lock` pins the complete environment: Torch 2.8.0+cu128,
Triton 3.4.0, Transformers 5.13.0 and W&B 0.31.0.

```bash
source /home/param/pose-repo/finetune/transcript_pose/environment/activate.sh
wandb login  # only needed on landsend-0
```

`bootstrap.sh` recreates packages using uv and downloads the exact FA3 kernel
revision `905da40048cf96bb967d35c5bea4bf38a715ace0`, build
`torch28-cxx11-cu128-x86_64-linux`. The HF repository now has type `kernel`.
No external FlashAttention package is needed. `PPM_FA3_ROOT` overrides the
archived trainer's kernel path while preserving its original default.

The base image lacks Python.h. Matching Ubuntu development headers are extracted
inside the venv; activation sets CPATH for Triton's driver compilation. No system
Torch, CUDA or Python packages are modified. The 595.91.07 host driver supports
the packaged CUDA 12.8 runtime. All 56 GPUs are H100 80GB HBM3.

The cluster uses IPv6 bootstrap on ibs14 and native InfiniBand/GDRDMA on all eight
mlx5 HCAs, automatically discovered. The launcher exports:

```bash
export NCCL_SOCKET_IFNAME='=ibs14'
export NCCL_SOCKET_FAMILY=AF_INET6
export GLOO_SOCKET_IFNAME=ibs14
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
```

The landsend-0 master address is `fdf9:c316:93ee:0:5f4:1538:2b0b:11d8`.
Using the explicit address avoids each hostname's loopback alias. Kernel
forward/backward checks passed on all 56 GPUs; a 56-rank all-reduce and FSDP2
gradient comparison passed before launch. Validation artifacts stay local.
