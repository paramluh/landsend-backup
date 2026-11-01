#!/usr/bin/env bash
# Source this file from bash before running the trainer.
POSE_ENV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
POSE_REPO_ROOT=$(cd -- "$POSE_ENV_DIR/../../.." && pwd)
source "$POSE_REPO_ROOT/.venv/bin/activate"
export PPM_FA3_ROOT=${PPM_FA3_ROOT:-$POSE_REPO_ROOT/.cache/kernels/vllm-flash-attn3}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$POSE_REPO_ROOT/.cache/triton}
export CPATH="$POSE_REPO_ROOT/.venv/include/python3.12:$POSE_REPO_ROOT/.venv/include${CPATH:+:$CPATH}"
export PYTHONUNBUFFERED=1
