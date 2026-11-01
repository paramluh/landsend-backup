#!/usr/bin/env bash
set -euo pipefail
source /home/param/pose-mono-dcp227/ops/env.sh
cd "$GENWEB"
export CUDA_VISIBLE_DEVICES=6,7
exec "$GENWEB_PYTHON" -u worker.py mono-nebius
