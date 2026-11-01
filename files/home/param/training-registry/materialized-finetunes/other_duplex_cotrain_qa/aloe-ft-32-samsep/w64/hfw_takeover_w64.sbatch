#!/bin/bash
#SBATCH -J aloesamsep-hfw -p all -N1 -n1 -c 4 --mem=16G -t 08:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/hfw-takeover-%j-%N.log
# w64: after train job 28511's trainer exits (control/TRAIN_EXITED-28511) the GPU job is ended so the 16 hosts go to ppmvox; this CPU-only job
# (no GPUs) finishes this host's node-local export -> private HF uploads with the same hf_watch.sh (train job 28511), nice 19 + idle IO.
export HFW_TRAIN_JOB=28511 QQA_ENV=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/w64/run_w64.env
exec nice -n 19 ionice -c3 bash /data/home/amartyavarma/ppm/aloe-ft-32-samsep/hf_watch.sh
