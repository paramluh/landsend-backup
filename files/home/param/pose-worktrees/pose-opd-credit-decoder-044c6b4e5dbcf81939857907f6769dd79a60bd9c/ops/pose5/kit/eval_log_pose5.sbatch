#!/bin/bash
#SBATCH -J ppm811s2p-evallog -p all -w slinky-4 -N1 -n1 -c1 --mem=1G -t 3-00:00:00 -o /data/home/amartyavarma/ppm/pose5/kit/logs/eval_log-%j.out
# usage: MIN_JOB=<pose5 job id> END=<PPM_MAX_STEPS> MIN_STEP=17850 sbatch [-w <spare>] eval_log_pose5.sbatch   (env exported by sbatch)
exec python3 /data/home/amartyavarma/ppm/pose5/kit/eval_log_pose5.py
