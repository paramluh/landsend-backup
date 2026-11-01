#!/bin/bash
#SBATCH -p all -w slinky-4 -N1 -n1 -c 2 --gres=none --mem=16G -t 20 -J pmpo-tabs
#SBATCH -o /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/logs/tabs-%j.out
exec /data/home/amartyavarma/venvs/qwen-s2/bin/python -u /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/work/tabs_page.py
