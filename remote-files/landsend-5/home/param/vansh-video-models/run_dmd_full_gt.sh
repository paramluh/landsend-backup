#!/usr/bin/env bash
# Run once on each node with the same config, NNODES, MASTER_ADDR and MASTER_PORT.
set -euo pipefail
if [ "$#" -lt 1 ]; then
    echo "Usage: NNODES=1 SLOTS=8 $0 CONFIG [--set key=value ...] [--benchmark N --benchmark-skip K]" >&2
    exit 2
fi
REPO_DIR=$(cd "$(dirname "$0")" && pwd)
CONFIG=$1
shift
NNODES=${NNODES:-1}
SLOTS=${SLOTS:-8}
if [ "$NNODES" -gt 1 ]; then
    : "${MASTER_ADDR:?Set the rendezvous host address on every node}"
    : "${NODE_RANK:?Set a distinct zero-based rank for each node}"
else
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    NODE_RANK=${NODE_RANK:-0}
fi
MASTER_PORT=${MASTER_PORT:-29932}
TORCHRUN=${TORCHRUN:-$REPO_DIR/.venv/bin/torchrun}
if [ ! -x "$TORCHRUN" ]; then
    echo "Set TORCHRUN to the torchrun executable in the training environment: $TORCHRUN is unavailable" >&2
    exit 2
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
# Local, per-node compiler caches; worker synchronization uses Inductor locks.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-$USER-full-gt}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-$USER-full-gt}
cd "$REPO_DIR"
exec "$TORCHRUN" --nnodes="$NNODES" --nproc_per_node="$SLOTS" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    learn_dmd_full_gt.py --config "$CONFIG" --expected-world-size "$((NNODES * SLOTS))" "$@"
