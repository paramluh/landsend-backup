#!/bin/bash
#SBATCH -J rosie-gestures -p all -x slinky-9 -N1 -n6 -c1 --mem=36G -t 0-01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-pose-rosie/logs/gestures-%j.log
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
srun -n6 -c1 --cpu-bind=none bash -c '$PY -I /data/home/amartyavarma/ppm/aloe-pose-rosie/gestures.py $SLURM_PROCID 6'
