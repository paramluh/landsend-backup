#!/bin/bash
#SBATCH -J ppm811s2p-watch -p all -w slinky-4 -N1 -n1 -c1 --mem=2G -t 4-00:00:00 -o /data/home/amartyavarma/ppm/logs/ppm811s2p-watchdog-%j.log
# pose5 8,1,1 watchdog (= code-811/ppm811f_watchdog.sbatch on a SPARE: slinky-15 is now a training host). CPU only. Disable: touch control/ppm-v1/WATCHDOG811_OFF
# usage: ENVF=<env> NODEF=<hosts file> TARGET=<PPM_MAX_STEPS> MIN_STEP=17850 sbatch [-w <spare>] ppm811s2p_watchdog.sbatch
exec bash /data/home/amartyavarma/ppm/pose5/kit/ppm811s2p_watchdog_v2.sh
