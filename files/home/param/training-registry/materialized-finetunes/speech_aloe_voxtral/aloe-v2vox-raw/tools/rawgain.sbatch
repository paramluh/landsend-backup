#!/bin/bash
#SBATCH -J v2vraw-gain -p all -N1 -n1 -c 8 --mem=64G --gres=gpu:1 -t 0:40:00 -w slinky-4
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2vox-raw/logs/rawgain-%j.log
cd /tmp
export GOOGLE_APPLICATION_CREDENTIALS=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 S3_MIN_FRAMES=125 PPM_CD=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/ppm/aloe-v2vox-raw/tools/build_rawgain_mimidec.py
