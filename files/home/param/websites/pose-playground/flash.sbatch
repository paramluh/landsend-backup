#!/bin/bash
#SBATCH -J genweb-flash
#SBATCH -x slinky-9
#SBATCH -c 6
#SBATCH --mem=8G
#SBATCH -n 1
#SBATCH -t 7-00:00:00
#SBATCH -o logs/%x-%j.out
# 10-26 (main): the playground's flash + beep burner in its own job (CPU only; up to 4 ffmpeg burns at once; 6 CPUs: all slinky-18 had free on 10-26). Restart loop.
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
bash $CANOPY_ROOT/ppm/aloe-posedup-live/tools/gpu_guard.sh "playground flash" || { echo "GPU guard refused playground flash; exiting"; exit 3; }   # 10-27 main: never slinky-4 GPUs 6-7 / SM < 1500 MHz for live / user-facing work
G=${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}; PY=$CANOPY_ROOT/venvs/qwen-s2/bin/python
while true; do $PY -u $G/flash_worker.py; echo "flash worker exited rc=$? $(date)"; sleep 5; done
