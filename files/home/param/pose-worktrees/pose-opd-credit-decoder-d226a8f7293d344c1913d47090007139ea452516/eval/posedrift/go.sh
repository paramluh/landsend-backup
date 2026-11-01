#!/bin/bash
# posedrift worker: nice 19, all node cores (cpuset 0-63), 1 torch thread per proc; args = score_calls.py args
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_HOME=/data/home/amartyavarma/.cache/huggingface
exec timeout 1800 nice -n 19 taskset -c 0-63 /data/home/amartyavarma/venvs/qwen-s2/bin/python /data/home/amartyavarma/ppm/aloe-posedup-live/posedrift/score_calls.py "$@"
