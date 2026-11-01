#!/usr/bin/env bash
set -euo pipefail
source /home/param/websites/pose-playground/ops/environment.sh
cd "$GENWEB"
exec "$GENWEB_PYTHON" -u front.py --port 9205
