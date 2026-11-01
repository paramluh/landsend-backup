#!/usr/bin/env bash
set -euo pipefail
NODE_RANK=${1:?Pass a contiguous node rank from 0 to 6}
ENV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ENV_DIR/activate.sh"
export NCCL_SOCKET_IFNAME='=ibs14'
export NCCL_SOCKET_FAMILY=AF_INET6
export GLOO_SOCKET_IFNAME=ibs14
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
PROBE_LOG_DIR=${PROBE_LOG_DIR:-$POSE_REPO_ROOT/.cache/nccl-probe}
mkdir -p "$PROBE_LOG_DIR"
export NCCL_DEBUG_FILE="$PROBE_LOG_DIR/nccl.%h.%p.log"
export POSE_NCCL_PROBE_OUTPUT="$ENV_DIR/validation/nccl-56-ranks.json"
exec timeout --signal=TERM --kill-after=20s 300s \
    "$POSE_REPO_ROOT/.venv/bin/torchrun" \
    --nnodes=7 --nproc_per_node=8 --node_rank="$NODE_RANK" \
    --master_addr=fdf9:c316:93ee:0:5f4:1538:2b0b:11d8 --master_port=29461 \
    --max_restarts=0 "$ENV_DIR/nccl_probe.py"
