#!/bin/bash
#SBATCH -J ppm811f-watch -p all -w slinky-15 -N1 -n1 -c1 --mem=2G -t 4-00:00:00 -o /data/home/amartyavarma/ppm/logs/ppm811f-watchdog-%j.log
# Cluster-side 8,1,1 watchdog (no Claude session needed). CPU only. Logic + knobs: ppm811f_watchdog.sh. Disable: touch control/ppm-v1/WATCHDOG811_OFF
exec bash /data/home/amartyavarma/ppm/pose5/code-811-s2p/ppm811f_watchdog.sh
