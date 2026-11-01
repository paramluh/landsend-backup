#!/bin/bash
#SBATCH -J aloev2 -p all -N8 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=5 --mem=540G -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2-ft/logs/aloev2-%j.log
# aloe-v2-ft (10-16 eve): = aloe-ft's aloe.sbatch (job 21980) + (1) a pre-flight that verifies / restages the 4 Model-1 stores on every
# node (= aloe-ft/stage.sbatch, digests asserted), (2) holds the nodes until control/GO exists (code finalised), then runs run_train.sh
# (= 21980's training block). Stop = save-and-exit: touch aloe-v2-ft/control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>).
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-v2-ft
echo "== aloe-v2 holder job $SLURM_JOB_ID on $(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,) $(TZ=America/Los_Angeles date '+%F %T PT')"
srun -N8 -n8 --ntasks-per-node=1 -c 5 --mem=32G --gres=none bash $Q/stage_check.sh || { echo "ABORT: store staging failed"; exit 1; }
for i in $(seq 1440); do [ -f $Q/control/GO ] && break; [ -f $Q/control/ABORT ] && { echo "== ABORT marker"; exit 1; }; sleep 10; done
[ -f $Q/control/GO ] || { echo "== no GO within 4 h"; exit 1; }
echo "== GO $(TZ=America/Los_Angeles date '+%T PT')"
exec bash $Q/run_train.sh
