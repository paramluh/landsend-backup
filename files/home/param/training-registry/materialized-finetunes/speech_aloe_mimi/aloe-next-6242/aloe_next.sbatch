#!/bin/bash
#SBATCH -J aloe-next-6242 -p all -N8 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=5 --mem=540G -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-next-6242/logs/aloe-next-6242-%j.log
#SBATCH --signal=B:USR1@300
# aloe-next-6242 (10-18) = aloe-v2init-8869/aloe_v2init.sbatch: store pre-flight (verify / restage the 4 Model-1 stores, digests asserted), GPU
# clock check, then run_train.sh. Stop = save-and-exit: touch control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>); never a bare scancel.
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-next-6242
echo "== aloe-next-6242 job $SLURM_JOB_ID on $(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,) $(TZ=America/Los_Angeles date '+%F %T PT')"
srun -N8 -n8 --ntasks-per-node=1 -c 5 --mem=32G --gres=none bash $Q/stage_check.sh || { echo "ABORT: store staging failed"; exit 1; }
srun -N8 -n8 --ntasks-per-node=1 -c 1 --mem=1G --gres=none bash -c 'echo "$(hostname -s) scratch $(df -h /scratch | tail -1)"'
srun -N8 -n8 --ntasks-per-node=1 -c 1 --mem=1G --gres=gpu:8 bash -c 'nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown --format=csv,noheader | sed "s/^/$(hostname -s) gpu /"' | sort -V
exec bash $Q/run_train.sh
