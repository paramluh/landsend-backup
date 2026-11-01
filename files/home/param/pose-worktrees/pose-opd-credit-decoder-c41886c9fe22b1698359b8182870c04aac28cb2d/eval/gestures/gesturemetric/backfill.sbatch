#!/bin/bash
#SBATCH -J gm-backfill
#SBATCH -w slinky-4
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH --gres=none
#SBATCH -t 3-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/agents1026/gesturemetric/logs/backfill-%j.out
# successor of the gm-cpu holder's backfill step: keeps writing runs/<id>/gesture_stats.json for new playground runs
OMP_NUM_THREADS=1 exec nice -n 19 /data/home/amartyavarma/venvs/qwen-s2/bin/python /data/home/amartyavarma/ppm/agents1026/gesturemetric/gm_backfill.py --loop 30
