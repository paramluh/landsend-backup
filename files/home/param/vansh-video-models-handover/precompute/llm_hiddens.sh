#!/usr/bin/env bash
# usage (on a node, from the repo): bash precompute/llm_hiddens.sh ROOT OUT WEIGHTS LOGS CSV...
# One precompute.llm_hiddens process per GPU, sharded by GPU index; returns at once, logs to LOGS/gpu<i>.log.
ROOT=$1 OUT=$2 WEIGHTS=$3 LOGS=$4
shift 4
cd "$(dirname "$0")/.."
mkdir -p "$LOGS"
N=$(nvidia-smi -L | wc -l)
for i in $(seq 0 $((N - 1))); do
    CUDA_VISIBLE_DEVICES=$i nohup .venv/bin/python -m precompute.llm_hiddens --root "$ROOT" --out "$OUT" \
        --weights "$WEIGHTS" --csvs "$@" --shard "$i" --num-shards "$N" > "$LOGS/gpu$i.log" 2>&1 &
done
echo "$(hostname): $N processes, logs in $LOGS"
