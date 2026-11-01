#!/bin/bash
#SBATCH -J aloev2pm17798 -p all -N8 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-main-17798/logs/aloev2pm17798-%j.log
# aloe-v2pure-main-17798 (MONOFT): = aloe-v2pure-17641/run.sbatch (job 27622) with this run's name / dirs; init = the main run's save at its pause
# (step 17798), node-local on the 8 FT hosts. Submitted by ppm/monoft/launch.sh (afterok on the init distribution job):
#   FTB_FINAL_SAVE=<0|1> sbatch -w <8 hosts> run.sbatch
# Stop: touch control/STOP_AFTER_SAVE. Resume: the same command again (same hosts; the node-local init must still be there).
exec bash /data/home/amartyavarma/ppm/aloe-v2pure-main-17798/run.sh
