#!/bin/bash
#SBATCH -J s2p-trainc -p all -w slinky-18 -x slinky-9 -N1 -n1 -c 8 --mem=48G -t 0-01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/pose-s2p/tests/logs/test_train_path-%j.log
# pose-s2p test (c): CPU only (no GPU requested); the mono8020 env (data cfg) + dummy trainer identity vars (no dist, nothing written to /data
# except the test's own json/log under pose-s2p/tests).
set -a; source /data/home/amartyavarma/ppm/aloe-pose-mono-8020/aloe_mono8020.env; set +a
export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 RANK=0 WORLD_SIZE=64 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8
S=/scratch/amartyavarma/pose-s2p-testc; mkdir -p $S
export QQA_RUN=s2p-testc QQA_LOG_ROOT=$S/runs QQA_PLAN_DIR=$S/plan QQA_CONTROL_DIR=$S/control QQA_EXPORT=$S/export ALOE_LOCAL_EXPORT_ROOT=$S/exports ALOE_SAVE_ROOT=$S/saves
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
echo "host $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT')"
$PY -u /data/home/amartyavarma/ppm/pose-s2p/tests/test_train_path.py
echo "rc $? $(TZ=America/Los_Angeles date '+%T PT')"
