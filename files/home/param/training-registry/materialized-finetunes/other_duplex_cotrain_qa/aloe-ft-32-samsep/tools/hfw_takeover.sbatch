#!/bin/bash
#SBATCH -J aloesamsep-hfw -p all -N1 -n1 -c 4 --mem=16G -t 08:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/hfw-takeover-%j-%N.log
# 10-24: FT 28249 stopped on main's order (GPUs needed for asr16861). The trainer has exited (control/TRAIN_EXITED-28249); this CPU-only job
# (no GPUs) takes over the node-local export -> private HF uploads from 28249's in-job watcher (same hf_watch.sh, train job 28249), nice 19 + idle IO.
export HFW_TRAIN_JOB=28249
exec nice -n 19 ionice -c3 bash /data/home/amartyavarma/ppm/aloe-ft-32-samsep/hf_watch.sh
