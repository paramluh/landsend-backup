#!/bin/bash
#SBATCH -J zoomscan -p all -N1 -n1 -c 8 --mem=32G --gres=none -t 00:40:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/scan_zoom-%j.log
# render + table + pack plan of zoom-strangers-5k (CPU only, compute node). usage: sbatch -w <spare node> scan_zoom.sbatch
N=/data/home/amartyavarma/ppm/aloe-zoom-cotrain
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
cd /tmp
nice -n 10 /data/home/amartyavarma/venvs/qwen-s2/bin/python -I $N/tools/scan_zoom.py $N/code $N/plan 8
