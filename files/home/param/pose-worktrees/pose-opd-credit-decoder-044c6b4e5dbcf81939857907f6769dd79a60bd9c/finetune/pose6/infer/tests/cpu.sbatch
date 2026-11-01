#!/bin/bash
#SBATCH -J pose6-a1-cpu -p all -x slinky-9 -N1 -n1 -c 8 --mem=48G -t 0-00:45:00
#SBATCH -o /data/home/amartyavarma/ppm/pose6/infer/tests/logs/cpu-%j.log
# pose6 A1 CPU checks (NO GPU): (1) cpu_load.py = engine load path (synthetic ckpts on node /scratch only), real key sets, inference SLOT;
# (2) cpu_slot_train.py = the FT trainer's P - 1 on real Aloe eval items + the parity micros. Writes only pose6/infer/tests/{out,logs} and
# /scratch/amartyavarma/pose6-a1/<job> (tiny synthetic test checkpoints; never /data).
# submit: sbatch [-w <node main assigns>] /data/home/amartyavarma/ppm/pose6/infer/tests/cpu.sbatch
set -u
echo "host $(hostname -s) start $(TZ=America/Los_Angeles date '+%F %T PT') job ${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
export A1_SCRATCH=/scratch/amartyavarma/pose6-a1/${SLURM_JOB_ID}
mkdir -p "${A1_SCRATCH:?}"
(
  source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
  $PY -u /data/home/amartyavarma/ppm/pose6/infer/tests/cpu_load.py S,L1,L2
  echo "cpu_load rc $?"
)
(
  set -a; source /data/home/amartyavarma/ppm/aloe-pose-mono-8020/aloe_mono8020.env; set +a
  export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 RANK=0 WORLD_SIZE=64 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8
  S=/scratch/amartyavarma/pose6-a1/${SLURM_JOB_ID}/trainenv; mkdir -p "${S:?}"
  export QQA_RUN=pose6-a1-slot QQA_LOG_ROOT=$S/runs QQA_PLAN_DIR=$S/plan QQA_CONTROL_DIR=$S/control QQA_EXPORT=$S/export ALOE_LOCAL_EXPORT_ROOT=$S/exports ALOE_SAVE_ROOT=$S/saves
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
  /data/home/amartyavarma/venvs/qwen-s2/bin/python -u /data/home/amartyavarma/ppm/pose6/infer/tests/cpu_slot_train.py
  echo "cpu_slot_train rc $?"
)
echo "end $(TZ=America/Los_Angeles date '+%F %T PT')"
