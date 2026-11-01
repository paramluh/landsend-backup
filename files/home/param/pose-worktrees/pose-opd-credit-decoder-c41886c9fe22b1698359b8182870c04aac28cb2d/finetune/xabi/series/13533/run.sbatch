#!/bin/bash
#SBATCH -J aloev2p13533 -p all -N8 -w slinky-5,slinky-6,slinky-8,slinky-10,slinky-11,slinky-16,slinky-17,slinky-19 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-13533/logs/aloev2p13533-%j.log
# aloe-v2pure-13533 (10-20, user): PPM 32B (init PPM step 13533) monoplex Aloe pose FT on the Aloe pose release v2 (= SAM 8B v2pure's data),
# = aloe-v2pure-16610 (job 26089) verbatim except the init. Stop: touch control/STOP_AFTER_SAVE. Resume: sbatch this file again.
exec bash /data/home/amartyavarma/ppm/aloe-v2pure-13533/run.sh
