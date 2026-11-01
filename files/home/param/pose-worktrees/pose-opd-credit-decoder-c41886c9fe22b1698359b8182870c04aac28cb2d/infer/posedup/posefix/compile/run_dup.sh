#!/bin/bash
# posefix COMPILE duplex check inside job 30592 (slinky-4 GPUs 0-1): test_compile.py on 2 held-out convs
D=/data/home/amartyavarma/ppm/aloe-posedup-live
C=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs
source $D/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm-live
export TORCHINDUCTOR_CACHE_DIR=$D/posefix/compile/inductor-cache
echo "CVD=$CUDA_VISIBLE_DEVICES start $(date -u +%T)"
export SKIP_DEC=1
exec $PY -u $D/posefix/sim/test_compile.py /scratch/amartyavarma/aloe-posedup-live/export-498-dcp $D/posefix/compile/test_compile_$1.json 300 $C/c0.npz $C/c1.npz
