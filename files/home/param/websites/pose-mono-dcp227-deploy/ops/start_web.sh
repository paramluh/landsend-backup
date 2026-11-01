#!/usr/bin/env bash
set -euo pipefail
source /home/param/pose-mono-dcp227/ops/env.sh
cd "$GENWEB"
exec "$GENWEB_PYTHON" -u server.py --port 9205
