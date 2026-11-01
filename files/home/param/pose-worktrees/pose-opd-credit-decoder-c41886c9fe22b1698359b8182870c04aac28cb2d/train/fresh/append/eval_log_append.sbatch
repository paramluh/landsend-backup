#!/bin/bash
#SBATCH -J ppm811f-evallog -p all -w slinky-15 -N1 -n1 -c1 --mem=1G -t 1-00:00:00 -o /data/home/amartyavarma/ppm/code-811/fresh/logs/eval_log-%j.out
# usage: MIN_JOB=<append job id> END=<PPM_MAX_STEPS> sbatch eval_log_append.sbatch   (env exported by sbatch)
exec python3 /data/home/amartyavarma/ppm/code-811/fresh/append/eval_log_append.py
