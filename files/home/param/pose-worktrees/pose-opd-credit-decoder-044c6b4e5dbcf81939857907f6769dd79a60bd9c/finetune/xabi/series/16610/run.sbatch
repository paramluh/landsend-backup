#!/bin/bash
#SBATCH -J aloev2p16610 -p all -N8 -w slinky-0,slinky-1,slinky-2,slinky-3,slinky-9,slinky-12,slinky-13,slinky-14 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-16610/logs/aloev2p16610-%j.log
# aloe-v2pure-16610 (10-20, user): PPM 32B (init PPM step 16610) monoplex Aloe pose FT on the Aloe pose release v2 (= SAM 8B v2pure's data),
# = aloe-v2pure-ft (job 23051) verbatim on 8 nodes (world 64). Stop: touch control/STOP_AFTER_SAVE. Resume: sbatch this file again.
exec bash /data/home/amartyavarma/ppm/aloe-v2pure-16610/run.sh
