#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
POSE_REPO_ROOT=$(cd -- "$ENV_DIR/../../.." && pwd)
POSE_UV=${POSE_UV:-/home/param/.local/bin/uv}
POSE_PYTHON=${POSE_PYTHON:-/usr/bin/python3.12}
if [[ ! -x "$POSE_UV" ]]; then
    echo "Install uv 0.13.6 in /home/param/.local/bin or set POSE_UV." >&2
    exit 1
fi
if [[ ! -x "$POSE_REPO_ROOT/.venv/bin/python" ]]; then
    "$POSE_UV" venv --python "$POSE_PYTHON" "$POSE_REPO_ROOT/.venv"
fi
"$POSE_UV" pip sync --python "$POSE_REPO_ROOT/.venv/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match "$ENV_DIR/requirements.lock"
# The base node image lacks Python.h. Keep matching Ubuntu 24.04 headers
# inside the venv so Triton can compile its tiny Python driver extension.
if [[ ! -f "$POSE_REPO_ROOT/.venv/include/python3.12/Python.h" ]]; then
    mkdir -p "$POSE_REPO_ROOT/.cache/python-headers"
    (
        cd "$POSE_REPO_ROOT/.cache/python-headers"
        apt-get download libpython3.12-dev=3.12.3-1ubuntu0.17
        dpkg-deb -x libpython3.12-dev_3.12.3-1ubuntu0.17_amd64.deb extracted
        cp -a extracted/usr/include/. "$POSE_REPO_ROOT/.venv/include/"
    )
fi
export PPM_FA3_ROOT=${PPM_FA3_ROOT:-$POSE_REPO_ROOT/.cache/kernels/vllm-flash-attn3}
"$POSE_REPO_ROOT/.venv/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="kernels-community/vllm-flash-attn3",
    repo_type="kernel",
    revision="905da40048cf96bb967d35c5bea4bf38a715ace0",
    allow_patterns=["build/torch28-cxx11-cu128-x86_64-linux/**"],
    local_dir=os.environ["PPM_FA3_ROOT"],
)
PY
"$POSE_UV" pip check --python "$POSE_REPO_ROOT/.venv/bin/python"
