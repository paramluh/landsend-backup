#!/bin/bash
# 10-27 main: GPU guard for LIVE / USER-FACING GPU services (see gpu_guard.py). First line of every such launch script:
#   bash /data/home/amartyavarma/ppm/aloe-posedup-live/tools/gpu_guard.sh "<service>" || exit 3
# exit 3 = refused (slinky-4 GPUs 6-7, or under a 3 s load: thermal/hw slowdown, >= 78 C, or SM < 900 MHz). GPU_GUARD_OFF=1 skips it (non-latency work only).
[ "${GPU_GUARD_OFF:-0}" = 1 ] && { echo "gpu guard: skipped (GPU_GUARD_OFF=1) for $1"; exit 0; }
exec /data/home/amartyavarma/venvs/qwen-s2/bin/python -u /data/home/amartyavarma/ppm/aloe-posedup-live/tools/gpu_guard.py "$1"
