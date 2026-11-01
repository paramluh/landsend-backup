#!/bin/bash
#SBATCH -J ppm811-test -p all --ntasks-per-node=8 --cpus-per-task=8 --gres=gpu:8 --mem=560G -t 0-01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/logs/ppm811test/evalref-%j.log
# 8,1,1 TEST (not a training run): the PRODUCTION trainer (ppm/code, read-only) resumed from a PPM node-local dcp with PPM_MAX_STEPS = that
# step -> it loads (digests asserted), runs evaluate(start) and exits: no optimizer step, no save, no export. The step-0 eval of the 8,1,1
# trainer at the same weights is compared against this (test811/README). Logs / metrics go to test dirs only (never runs/ppm-v1-*).
# usage: STEP=16762 sbatch -N 16 -w /data/home/amartyavarma/ppm/code/nodes16.txt evalref_ppm.sbatch   [CODE=<code dir> to run another copy]
set -uo pipefail
C=${CODE:-/data/home/amartyavarma/ppm/code}; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source /data/home/amartyavarma/ppm/code/prod_ppm.env; set +a
export PPM_RESUME_FROM=/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-$STEP PPM_MAX_STEPS=$STEP
export PPM_RUN_BASE=ppm811-test PPM_RUN=ppm811-evalref-${TAG:-prod}-j$SLURM_JOB_ID PPM_LOG_ROOT=/data/home/amartyavarma/ppm/runs-811test
export PPM_SAVE_ROOT=/scratch/amartyavarma/ppm811-test/ckpt PPM_CONTROL_DIR=/data/home/amartyavarma/ppm/control/ppm811-test
export PPM_FINAL_SAVE=0 PPM_SAVE_EVERY_MIN=0 PPM_EXPORT_EVERY_MIN=0 PPM_EVAL_EVERY_MIN=0 MASTER_PORT=29871
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
mkdir -p "$PPM_CONTROL_DIR"
echo "== ppm811-test evalref job $SLURM_JOB_ID on $PPM_HOSTS code $C resume $PPM_RESUME_FROM $(TZ=America/Los_Angeles date '+%F %T PT') run $PPM_RUN"
srun --mem=560G --ntasks-per-node=8 --cpus-per-task=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_ppm.py
echo "== exit $? $(TZ=America/Los_Angeles date '+%T PT')"
