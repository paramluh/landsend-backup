#!/bin/bash
# posefix COMPILE monoplex check: test_compile_mono.py on 2 held-out convs (B track teacher-forced, A empty)
D=/data/home/amartyavarma/ppm/aloe-posedup-live
C=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs
source $D/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm-live MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
export TORCHINDUCTOR_CACHE_DIR=$D/posefix/compile/inductor-cache
echo "CVD=$CUDA_VISIBLE_DEVICES start $(date -u +%T)"
exec $PY -u $D/posefix/sim/test_compile_mono.py /data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357 $D/posefix/compile/test_compile_mono_$1.json 200 $C/c4.npz $C/c5.npz
