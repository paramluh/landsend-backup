source "$(dirname "$0")/nodes.sh"

# usage: [CONFIG=...] bash run_multinode.sh {r|p|l|m} NODE...
# e.g.   bash run_multinode.sh p 1 2 3 4
LEARNER=learn_bimodal.py
CONFIG=${CONFIG:-experiments/bimodal/aloe-studio-bimodal-studio.toml}
NODES_TO_USE=("${@:2}")
SLOTS=8
MASTER_PORT=29521

CLUSTER=$(resolve_cluster "$1") || exit 1
if [[ ${#NODES_TO_USE[@]} -eq 0 ]]; then
    echo "usage: bash run_multinode.sh {r|p|l|m} NODE..." >&2
    exit 1
fi

declare -n NODES="NODES_${CLUSTER}"
MASTER_ADDR="${NODES[${NODES_TO_USE[0]}]}"
IFNAME=eth0
# landsend, richmond: ens1 is NAT (same address on every node); nodes reach each other by hostname over IPoIB.
if [[ $CLUSTER == LANDSEND || $CLUSTER == RICHMOND ]]; then
    MASTER_ADDR=${CLUSTER,,}-${NODES_TO_USE[0]}
    IFNAME=ibs14
fi
# richmond: the public IPs are unreachable from inside the cluster, so ssh goes by hostname too.
host() {
    if [[ $CLUSTER == RICHMOND ]]; then echo "richmond-$1"; else echo "${NODES[$1]}"; fi
}
NUM_NODES=${#NODES_TO_USE[@]}
REPO_DIR=$(cd "$(dirname "$0")" && pwd)

cleanup() {
    for n in "${NODES_TO_USE[@]}"; do
        ssh "$(host "$n")" "pkill -f '$LEARNER --config'" 2>/dev/null
    done
}
trap cleanup INT TERM

for i in "${!NODES_TO_USE[@]}"; do
    ssh "$(host "${NODES_TO_USE[$i]}")" "
        cd '$REPO_DIR' && ulimit -n \$(ulimit -Hn) &&
        export NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 &&
        .venv/bin/torchrun \
            --nnodes=$NUM_NODES --node_rank=$i --nproc_per_node=$SLOTS \
            --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
            $LEARNER --config $CONFIG
    " 2>&1 &
done

wait
