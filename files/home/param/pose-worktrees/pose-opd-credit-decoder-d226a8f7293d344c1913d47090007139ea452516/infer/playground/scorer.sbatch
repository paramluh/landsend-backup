#!/bin/bash
#SBATCH -J genweb-scorer
#SBATCH -x slinky-9
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 7-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/%x-%j.out
# 10-26 (user via main): the playground's pose-stats scorer in its OWN job (CPU only), so it can never push the web job (4 GB, holds the
# public tunnel) over its memory limit. Writes runs/<id>/pose_stats.json; restart loop.
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while true; do $PY -u $G/pose_scorer.py; echo "scorer exited rc=$? $(date)"; sleep 10; done
