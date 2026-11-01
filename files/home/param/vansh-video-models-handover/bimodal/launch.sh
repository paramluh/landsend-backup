source "$(dirname "$0")/../nodes.sh"

# usage: [GPUS="0 1 2 3"] bash bimodal/launch.sh SCRIPT YAML NODE...   (reserved Nebius nodes; one SCRIPT shard per GPU)
# e.g.   bash bimodal/launch.sh bimodal/sampler.py bimodal/sampler.yaml 0 1
SCRIPT=$1
YAML=$2
NODES_TO_USE=("${@:3}")
GPU_LIST=(${GPUS:-0 1 2 3 4 5 6 7})
SLOTS=${#GPU_LIST[@]}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
NUM_SHARDS=$((SLOTS * ${#NODES_TO_USE[@]}))

cleanup() {
    for n in "${NODES_TO_USE[@]}"; do
        ssh "${NODES_RESERVED_NEBIUS[$n]}" "pkill -f '$SCRIPT $YAML'" 2>/dev/null
    done
}
trap cleanup INT TERM

for i in "${!NODES_TO_USE[@]}"; do
    ssh "${NODES_RESERVED_NEBIUS[${NODES_TO_USE[$i]}]}" "
        cd '$REPO_DIR' && g=0 && for gpu in ${GPU_LIST[*]}; do
            CUDA_VISIBLE_DEVICES=\$gpu .venv/bin/python $SCRIPT $YAML --shard \$(($i * $SLOTS + g)) --num-shards $NUM_SHARDS &
            g=\$((g + 1))
        done; wait
    " 2>&1 &
done

wait
