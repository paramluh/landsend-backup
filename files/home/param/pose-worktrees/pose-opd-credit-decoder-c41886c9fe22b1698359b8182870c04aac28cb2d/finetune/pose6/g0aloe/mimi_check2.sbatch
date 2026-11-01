#!/bin/bash
#SBATCH -J pose6-mimichk2 -p all -N1 -n1 -c 4 --mem=24G -t 0:20:00 -w slinky-18 --nice=100
#SBATCH -o /data/home/amartyavarma/ppm/pose6/g0aloe/logs/mimichk2-%j.log
# pose6 B2: Mimi recipe census + cb0 histogram spot check (CPU, read-only inputs; writes only pose6/g0aloe/mimi_check.json).
echo "== $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT')"
export OMP_NUM_THREADS=1
nice -n 10 /data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/ppm/pose6/g0aloe/mimi_check2.py
echo "== exit $? $(TZ=America/Los_Angeles date '+%T PT')"
