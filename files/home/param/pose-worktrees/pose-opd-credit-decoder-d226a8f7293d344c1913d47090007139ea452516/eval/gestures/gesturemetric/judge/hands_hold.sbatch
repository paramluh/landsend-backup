#!/bin/bash
#SBATCH -J gm-judge-hands
#SBATCH -w slinky-10
#SBATCH -x slinky-9
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 3:00:00
#SBATCH -o /data/home/amartyavarma/ppm/agents1026/gesturemetric/judge/logs/hands-%j.out
# hands/arms-only judge retrain (main 10-27): holds slinky-10 GPUs, runs judge/hands_run.sh once it exists; exits when that ends
J=/data/home/amartyavarma/ppm/agents1026/gesturemetric/judge
echo "$(date -u +%H:%M:%S) holder up on $(hostname) GPUs=$CUDA_VISIBLE_DEVICES"
for i in $(seq 1 240); do [ -f $J/hands_run.sh ] && break; sleep 15; done
[ -f $J/hands_run.sh ] || { echo "no hands_run.sh after 60 min, releasing"; exit 1; }
echo "$(date -u +%H:%M:%S) running hands_run.sh"
bash $J/hands_run.sh
echo "$(date -u +%H:%M:%S) hands_run.sh exit $?"
