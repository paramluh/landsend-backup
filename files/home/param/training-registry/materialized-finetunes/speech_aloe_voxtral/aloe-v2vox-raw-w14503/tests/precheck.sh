#!/bin/bash
# CPU pre-launch checks for aloe-v2vox-raw-w14503 (init completeness + resolved-config diff vs 23081). Run inside a CPU slurm step.
set -a; source /data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/aloe_v2vox_raw.env; set +a
export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-14503 QQA_RUN=ppm-aloe-v2vox-raw-w14503-j0 RANK=0 WORLD_SIZE=24 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8 ALOE_LNODES="0,1,2;3,4,5;6,7" OMP_NUM_THREADS=4
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh >/dev/null 2>&1
cd /tmp
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/tests/precheck.py
