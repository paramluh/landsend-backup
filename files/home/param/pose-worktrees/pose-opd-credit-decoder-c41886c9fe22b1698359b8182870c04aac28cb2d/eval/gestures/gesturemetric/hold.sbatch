#!/bin/bash
#SBATCH -J gm-cpu
#SBATCH -w slinky-4
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --gres=none
#SBATCH -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/agents1026/gesturemetric/logs/hold-%j.out
while [ ! -f /data/home/amartyavarma/ppm/agents1026/gesturemetric/STOP_HOLD ]; do sleep 30; done
