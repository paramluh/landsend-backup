#!/bin/bash
# re-attach the HF watcher (fixed hf_watch.sh: per-host NODE_COMPLETE not uploaded) as an overlap step of train job 28511 (the original step
# 28511.9 was cancelled 10-24 02:42 PT); run from a tiny CPU job so the srun client survives login-node restarts.
for v in $(env | grep -oE '^(SLURM|SBATCH|SRUN)_[A-Za-z0-9_]+'); do unset $v; done
export QQA_ENV=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/w64/run_w64.env HFW_TRAIN_JOB=28511
exec srun --overlap --jobid=28511 -N8 -n8 --ntasks-per-node=1 -c 2 --mem=16G --gres=none --cpu-bind=none --kill-on-bad-exit=0 \
  nice -n 10 bash /data/home/amartyavarma/ppm/aloe-ft-32-samsep/hf_watch.sh
