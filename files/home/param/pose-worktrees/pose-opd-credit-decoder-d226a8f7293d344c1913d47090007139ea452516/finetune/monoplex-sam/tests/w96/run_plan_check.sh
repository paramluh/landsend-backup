#!/bin/bash
# CPU plan check (plan_w96_check.py) on a compute node: srun -w slinky-18 -n1 -c4 --mem=32G bash run_plan_check.sh
set -uo pipefail
T=/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/tests/w96
set -a; source /data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/aloe_mono8020.env; set +a
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
exec /data/home/amartyavarma/venvs/qwen-s2/bin/python $T/plan_w96_check.py
