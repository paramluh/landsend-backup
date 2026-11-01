#!/bin/bash
#SBATCH -J aloev2i-8869 -p all -N8 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=5 --mem=540G -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2init-8869/logs/aloev2i-8869-%j.log
# aloe-v2init-8869 (10-17) = aloe-v2-ft/aloe_v2.sbatch (job 22383): store pre-flight (verify / restage the 4 Model-1 stores, digests asserted)
# then run_train.sh. No GO hold (code final). Stop = save-and-exit: touch control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>).
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-v2init-8869
echo "== aloe-v2init-8869 job $SLURM_JOB_ID on $(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,) $(TZ=America/Los_Angeles date '+%F %T PT')"
srun -N8 -n8 --ntasks-per-node=1 -c 5 --mem=32G --gres=none bash $Q/stage_check.sh || { echo "ABORT: store staging failed"; exit 1; }
srun -N8 -n8 --ntasks-per-node=1 -c 1 --mem=1G --gres=none bash -c 'echo "$(hostname -s) scratch $(df -h /scratch | tail -1)"'
exec bash $Q/run_train.sh
