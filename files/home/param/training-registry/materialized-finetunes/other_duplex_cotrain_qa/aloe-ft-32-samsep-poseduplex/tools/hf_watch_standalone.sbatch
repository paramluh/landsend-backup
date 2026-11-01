#!/bin/bash
#SBATCH -J pdx-hfwatch -p all -N1 -n1 -c2 --mem=64G -t 01:30:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/hf_watch-standalone-%j.log
# 10-25 (ppmguard, main -c9 go): finish pose-duplex job 30074's HF uploads in a small CPU-only job per host, so the trainer job can release its
# 8 x 8 GPUs. Same env the trainer sbatch gives hf_watch.sh; HFW_TRAIN_JOB=30074 -> its TRAIN_EXITED marker + hf_done/j30074 (resumes, skips done).
# usage: sbatch -w <host> --dependency=afterany:30074 tools/hf_watch_standalone.sbatch   (PRIVATE HF ONLY, NEVER GCS)
set -uo pipefail
export POSEDUP_HOSTS=slinky-1,slinky-2,slinky-6,slinky-15,slinky-7,slinky-13,slinky-14,slinky-17 QQA_STEPS_SET=498 HFW_TRAIN_JOB=30074
echo "== pdx-hfwatch job $SLURM_JOB_ID on $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT') (train job $HFW_TRAIN_JOB)"
nice -n 10 bash /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/hf_watch.sh
echo "== hf watcher exit $? $(TZ=America/Los_Angeles date '+%T PT')"
