#!/bin/bash
#SBATCH -J aloev2p17352 -p all -N8 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-17352/logs/aloev2p17352-%j.log
# aloe-v2pure-17352 (10-22, user via coordinator: v2pure scaling series, init PPM 8,1,1 step 17352) = aloe-v2pure-16610 (job 26089) verbatim except
# the init (+ no hourly save, run.env [SPACE]; final resumable save per FTB_FINAL_SAVE, run.sh); 8 of the 16 PPM hosts (world 64), passed at submit:
#   FTB_FINAL_SAVE=<0|1> sbatch -w <8 hosts> run.sbatch      (aloe-v2pure-17641/tools/ftb_chain.sh)
# Stop: touch control/STOP_AFTER_SAVE. Resume: the same command again (same hosts).
exec bash /data/home/amartyavarma/ppm/aloe-v2pure-17352/run.sh
