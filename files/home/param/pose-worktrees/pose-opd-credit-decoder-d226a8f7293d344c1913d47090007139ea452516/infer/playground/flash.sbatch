#!/bin/bash
#SBATCH -J genweb-flash
#SBATCH -x slinky-9
#SBATCH -c 6
#SBATCH --mem=8G
#SBATCH -n 1
#SBATCH -t 7-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/%x-%j.out
# 10-26 (main): the playground's flash + beep burner in its own job (CPU only; up to 4 ffmpeg burns at once; 6 CPUs: all slinky-18 had free on 10-26). Restart loop.
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
while true; do $PY -u $G/flash_worker.py; echo "flash worker exited rc=$? $(date)"; sleep 5; done
