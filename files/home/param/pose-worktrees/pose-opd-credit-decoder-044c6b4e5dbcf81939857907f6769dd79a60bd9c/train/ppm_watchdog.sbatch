#!/bin/bash
#SBATCH -J ppm-watchdog -p all -w slinky-15 -N1 -n1 -c1 --mem=2G -t 4-00:00:00 -o /data/home/amartyavarma/ppm/logs/ppm-watchdog-%j.log
# Cluster-side PPM watchdog (no Claude session needed). CPU only. Logic + knobs: ppm_watchdog.sh. Disable: touch control/ppm-v1/WATCHDOG_OFF
exec bash /data/home/amartyavarma/ppm/code/ppm_watchdog.sh
