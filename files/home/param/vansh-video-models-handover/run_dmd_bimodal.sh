# usage: bash run_dmd_bimodal.sh "NODE_IPS_OR_HOSTNAMES" CONFIG [learn_dmd_bimodal.py args...]
# Runs torchrun on every listed node (the first is the master) from this directory; logs to stdout.
set -euo pipefail
NODES=($1)
CONFIG=$2
shift 2
ARGS=''
if [ $# -gt 0 ]; then ARGS=$(printf '%q ' "$@"); fi
REPO_DIR=$(cd "$(dirname "$0")" && pwd)
source "$REPO_DIR/nodes.sh"
for i in "${!NODES[@]}"; do
    for n in "${!NODES_RICHMOND[@]}"; do
        if [[ ${NODES[$i]} == "${NODES_RICHMOND[$n]}" ]]; then NODES[$i]="richmond-$n"; fi
    done
done
DEFAULT_IFNAME=eth0
if [[ ${NODES[0]} == richmond-* ]]; then DEFAULT_IFNAME=ibs14; fi
IFNAME=${IFNAME:-$DEFAULT_IFNAME}
MASTER_ADDR=${MASTER_ADDR:-${NODES[0]}}
MASTER_PORT=${MASTER_PORT:-29531}
SLOTS=${SLOTS:-8}
WANDB_ENV_FILE=${WANDB_ENV_FILE:-}
PIDS=()
for i in "${!NODES[@]}"; do
    ssh -o BatchMode=yes "${NODES[$i]}" "
        set -e; cd '$REPO_DIR'; ulimit -n \$(ulimit -Hn);
        if [ -f .env ]; then set -a; source .env; set +a; fi;
        if [ -n '$WANDB_ENV_FILE' ] && [ -f '$WANDB_ENV_FILE' ]; then source '$WANDB_ENV_FILE'; fi;
        export NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
               TORCHINDUCTOR_CACHE_DIR=/tmp/param-dmd-inductor TRITON_CACHE_DIR=/tmp/param-dmd-triton OMP_NUM_THREADS=4;
        .venv/bin/torchrun --nnodes=${#NODES[@]} --node_rank=$i --nproc_per_node=$SLOTS \
            --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
            learn_dmd_bimodal.py --config $CONFIG $ARGS
    " 2>&1 &
    PIDS+=("$!")
done
STATUS=0
for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
exit "$STATUS"
