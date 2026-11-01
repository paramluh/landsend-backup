#!/usr/bin/env bash
set -euo pipefail
rank=${1:?rank}
hosts=(landsend-1 landsend-2 landsend-3 landsend-4 landsend-5 landsend-12)
[[ "$rank" =~ ^[0-5]$ && "$(hostname -s)" == "${hosts[$rank]}" ]] || exit 2
source /home/param/pose-repo/finetune/transcript_pose/environment/activate.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME='=ibs14' NCCL_SOCKET_FAMILY=AF_INET6 GLOO_SOCKET_IFNAME=ibs14
export NCCL_IB_DISABLE=0 NCCL_NET=IB TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export NCCL_DEBUG_FILE=/home/param/pose-runs/duplex-distill-ops-20261029/nccl48/nccl.%h.%p.log
export POSE_NCCL_PROBE_OUTPUT=/home/param/pose-runs/duplex-distill-ops-20261029/nccl48/result.json
exec timeout --signal=TERM --kill-after=10s 150s /home/param/pose-repo/.venv/bin/torchrun --nnodes=6 --nproc_per_node=8 --node_rank="$rank" --master_addr=fdf9:c316:93ee:0:5f4:1564:2f9a:6448 --master_port=29574 --max_restarts=0 /home/param/pose-repo/finetune/transcript_pose/environment/nccl_probe.py
