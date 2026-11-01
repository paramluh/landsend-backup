#!/bin/bash
#SBATCH -J aloesamsep-release -p all -N1 -n1 -c 1 --mem=512M -t 06:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/w64-release-%j.log
# w64: free the GPUs the moment the trainer of job 29444 has exited (run_w64.sbatch writes control/TRAIN_EXITED-29444 after every rank's final
# export / stop export is on disk); the job would otherwise hold all 64 GPUs while its in-job HF watcher uploads. Uploads continue in the
# CPU-only aloesamsep-hfw jobs (afterany:29444). Never cancels a job whose trainer is still running.
C=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/control; J=29444
while squeue -h -j $J | grep -q .; do
  if [ -f $C/TRAIN_EXITED-$J ]; then echo "[$(TZ=America/Los_Angeles date '+%T PT')] trainer exited rc $(cat $C/TRAIN_EXITED-$J): scancel $J (GPU release)"; sleep 5; scancel $J; break; fi
  sleep 10
done
echo "[$(TZ=America/Los_Angeles date '+%T PT')] job $J gone"
