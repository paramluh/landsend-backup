#!/bin/bash
#SBATCH -p all -w slinky-4 -N1 -n1 -c 2 --gres=none --mem=24G -t 30 -J pmpo-stpage
#SBATCH -o /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/logs/stpage-%j.out
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PPM_CD=2 OMP_NUM_THREADS=2
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
exec /data/home/amartyavarma/venvs/qwen-s2/bin/python -u /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/work/final_studio_page.py
