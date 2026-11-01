#!/bin/bash
#SBATCH -J aloev2pureft -p all -N3 -w slinky-4,slinky-7,slinky-15 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-ft/logs/aloev2pureft-%j.log
# aloe-v2pure-ft (10-17): PPM 32B monoplex Aloe pose fine-tune (= job 22317's recipe) on the Aloe pose release v2 (the 8B SAM pose v2pure data).
# 3 spare nodes x 8 H100 (world 24). Stop: touch control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>). Resume: sbatch this file again.
exec bash /data/home/amartyavarma/ppm/aloe-v2pure-ft/aloe_v2pure_run.sh
