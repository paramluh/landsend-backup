#!/usr/bin/env bash
# Five-node resolver training; each launch owns only its torchrun processes.
set -euo pipefail
read -r -a NODES <<< "$1"
CONFIG=$2
shift 2
REPO_DIR=$(cd "$(dirname "$0")" && pwd)
SLOTS=${SLOTS:-8}
MASTER_ADDR=${MASTER_ADDR:-${NODES[0]}}
MASTER_PORT=${MASTER_PORT:-29921}
IFNAME=${IFNAME:-ibs14}
WANDB_ENV_FILE=${WANDB_ENV_FILE:-/home/param/.config/stabilizer-dmd/wandb.env}
RESOLVER_RUN_TAG=${RESOLVER_RUN_TAG:-resolver-$(date -u +%Y%m%d-%H%M%S)}
QUOTED_ARGS=''
if [ "$#" -gt 0 ]; then printf -v QUOTED_ARGS '%q ' "$@"; fi
printf -v QUOTED_CONFIG '%q' "$CONFIG"
printf -v QUOTED_REPO '%q' "$REPO_DIR"
PIDS=()
for i in "${!NODES[@]}"; do
    ssh -o BatchMode=yes "${NODES[$i]}" "
        set -e; cd $QUOTED_REPO; ulimit -n \$(ulimit -Hn);
        if [ -f /home/param/vansh-video-models/.env ]; then set -a; source /home/param/vansh-video-models/.env; set +a; fi;
        if [ -f '$WANDB_ENV_FILE' ]; then source '$WANDB_ENV_FILE'; fi;
        export NCCL_SOCKET_IFNAME='$IFNAME' GLOO_SOCKET_IFNAME='$IFNAME' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
          TORCHINDUCTOR_CACHE_DIR=/tmp/param-dmd-inductor TRITON_CACHE_DIR=/tmp/param-dmd-triton OMP_NUM_THREADS=4 \
          RESOLVER_RUN_TAG='$RESOLVER_RUN_TAG';
        /home/param/vansh-video-models/.venv/bin/torchrun --nnodes=${#NODES[@]} --node_rank=$i --nproc_per_node=$SLOTS \
          --master_addr='$MASTER_ADDR' --master_port='$MASTER_PORT' learn_dmd_resolver.py --config $QUOTED_CONFIG $QUOTED_ARGS
    " 2>&1 &
    PIDS+=("$!")
done
STATUS=0
for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
exit "$STATUS"
