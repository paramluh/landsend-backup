#!/bin/bash
#SBATCH -J c1-probe -p all -w slinky-18 -N1 -n1 -c 2 --mem=8G --nice=0 -t 0-00:20:00 --gres=none
#SBATCH -o /data/home/amartyavarma/ppm/pose6/mono17850/convert/logs/probe-%j.log
D=/data/home/amartyavarma/ppm/pose6/mono17850/convert
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I -u $D/probe_hf.py
