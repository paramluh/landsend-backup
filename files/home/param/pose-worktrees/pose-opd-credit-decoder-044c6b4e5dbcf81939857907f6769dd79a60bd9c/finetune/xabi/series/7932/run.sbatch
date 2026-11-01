#!/bin/bash
#SBATCH -J aloev2p7932 -p all -N8 -w slinky-0,slinky-1,slinky-2,slinky-3,slinky-9,slinky-12,slinky-13,slinky-14 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-7932/logs/aloev2p7932-%j.log
# aloe-v2pure-7932 (10-20, user): PPM 32B (init PPM step 7932) monoplex Aloe pose FT on the Aloe pose release v2 (= SAM 8B v2pure's data),
# = aloe-v2pure-16610 (job 26089) verbatim except the init. Stop: touch control/STOP_AFTER_SAVE. Resume: sbatch this file again.
exec bash /data/home/amartyavarma/ppm/aloe-v2pure-7932/run.sh
