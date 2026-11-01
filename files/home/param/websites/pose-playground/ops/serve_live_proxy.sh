#!/usr/bin/env bash
set -euo pipefail
source /home/param/websites/pose-playground/ops/environment.sh
export PPMLIVE_AVATAR_UPSTREAM=http://127.0.0.1:9227
export PPMLIVE_AVATAR2_UPSTREAM=http://127.0.0.1:9227
export PPMLIVE_AVATAR3_UPSTREAM=http://127.0.0.1:9227
exec "$GENWEB_PYTHON" -u "$GENWEB_LIVE/proxy.py" --port 9215 --backend 9216 \
  --token-file "$GENWEB_LIVE_STATE/backend.token" --sessions "$GENWEB_LIVE_STATE/sessions" \
  --lb /home/param/runs/pose-runs/pose-sites-20261030/live-lb.json
