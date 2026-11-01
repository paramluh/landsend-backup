#!/bin/bash
# pose6 A3: CPU-only worker test (no GPU, no model load; ~2-4 min). Run on a COMPUTE node, never the login node:
#   sbatch /data/home/amartyavarma/ppm/pose6/worker/tests/test_worker_cpu.sbatch
#   or inside a job main grants:  srun --overlap --jobid <J> -w <node> -x slinky-9 -n 1 -c 4 --mem=32G --cpu-bind=none \
#        bash /data/home/amartyavarma/ppm/pose6/worker/tests/run_cpu_test.sh
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
unset GENWEB_CK GENWEB_WEIGHTS
echo "host $(hostname) $(date -u +%FT%TZ) py $PY"
$PY -B /data/home/amartyavarma/ppm/pose6/worker/tests/test_worker_cpu.py
rc=$?; echo "rc=$rc $(date -u +%FT%TZ)"; exit $rc
