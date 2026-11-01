#!/bin/bash
#SBATCH -J aloev2p17641 -p all -t 2-00:00:00 -o /data/home/amartyavarma/ppm/aloe-v2pure-17641/logs/aloev2p17641-%j.log
#SBATCH -N1 -w slinky-15 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G
#SBATCH hetjob
#SBATCH -N2 -w slinky-4,slinky-18 --ntasks-per-node=6 --cpus-per-task=4 --gres=gpu:6 --mem=380G
# aloe-v2pure-17641 SPARE FALLBACK (= aloe-v2pure-17352/run.sbatch, FT-(a), with the init 17641): PPM 32B (init PPM 8,1,1 step 17641) monoplex Aloe pose FT on the
# Aloe pose release v2 (= SAM 8B v2pure's data) = aloe-v2pure-16610 (job 26089) verbatim except the init (+ no hourly save, run.env [SPACE]).
# World 20 on the spare GPUs: slinky-15 x8 + slinky-4 GPUs 0-5 (6-7 throttled) + slinky-18 GPUs 2-7 (0-1 = the user's live demo job 27040);
# Slurm assigns slinky-4 IDX 0-5 / slinky-18 IDX 2-7 (checked 10-22); code/task_ppm.sh pins CUDA_VISIBLE_DEVICES to exactly those.
# Stop: touch control/STOP_AFTER_SAVE. Resume: sbatch this file again (same nodes / world).
exec bash /data/home/amartyavarma/ppm/aloe-v2pure-17641/run_spares.sh
