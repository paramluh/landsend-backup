#!/bin/bash
#SBATCH -J pose6-g0aloe -p all -N1 -n1 -c 4 --mem=16G -t 0:40:00 -w slinky-18 --nice=100
#SBATCH -o /data/home/amartyavarma/ppm/pose6/g0aloe/logs/build-%j.log
# pose6 B2: Aloe frame-0 G0COND sidecar (CPU only, read-only inputs; writes only pose6/g0aloe/).
echo "== $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT')"
export OMP_NUM_THREADS=1
nice -n 10 /data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/ppm/pose6/g0aloe/build_g0aloe.py --workers 4
echo "== exit $? $(TZ=America/Los_Angeles date '+%T PT')"
