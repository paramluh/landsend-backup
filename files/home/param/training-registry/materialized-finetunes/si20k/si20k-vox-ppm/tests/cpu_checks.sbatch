#!/bin/bash
#SBATCH -p all -N1 -n1 -c 16 --mem=64G --gres=none -t 1:00:00 -J si20k-cpucheck
# si20k-vox CPU checks (tests/cpu_checks.py) on a node holding the store set. usage: sbatch -w <node> -o LOG cpu_checks.sbatch <stores dir> <out json>
export PPM_CD=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 HF_HOME=/data/home/amartyavarma/.cache/huggingface
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh >/dev/null 2>&1
cd /tmp
nice -n 5 /data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/ppm/si20k-vox-ppm/tests/cpu_checks.py --stores $1 --out $2 --n 160
