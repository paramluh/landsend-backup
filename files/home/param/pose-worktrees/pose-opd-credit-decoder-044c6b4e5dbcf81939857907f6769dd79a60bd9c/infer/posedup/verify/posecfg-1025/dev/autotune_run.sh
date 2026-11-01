#!/bin/bash
# 4 separate processes (= 4 'backend restarts'): old, new, old, new; 2 GPUs each, 2 at a time on the 4 GPUs of the dev job
D=/data/home/amartyavarma/ppm/aloe-posedup-live/verify/posecfg-1025/dev
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-posecfg-dev
CUDA_VISIBLE_DEVICES=0,1 $PY -u $D/autotune_test.py old $D/ci498.pkl $D/at_old1.npz > $D/at_old1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 $PY -u $D/autotune_test.py new $D/ci498.pkl $D/at_new1.npz > $D/at_new1.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0,1 $PY -u $D/autotune_test.py old $D/ci498.pkl $D/at_old2.npz > $D/at_old2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 $PY -u $D/autotune_test.py new $D/ci498.pkl $D/at_new2.npz > $D/at_new2.log 2>&1 &
wait
echo ATDONE
