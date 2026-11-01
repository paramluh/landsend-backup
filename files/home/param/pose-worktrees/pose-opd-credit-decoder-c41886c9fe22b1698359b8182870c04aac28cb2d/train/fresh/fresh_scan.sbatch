#!/bin/bash
#SBATCH -p all -N1 -n1 -c 64 --mem=200G -t 1:00:00 -J fresh-scan
#SBATCH -o /data/home/amartyavarma/ppm/code-811/fresh/logs/scan-%N-%j.out
# fresh pose (8,1,1 resume): this host's scan (release hash verification + real decode filter) -> global dedup -> this host's fixed row plan
rm -rf /scratch/amartyavarma/ppm-fresh-pose-test
exec /data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/ppm/code-811/fresh/fresh_pose.py all --workers 60
