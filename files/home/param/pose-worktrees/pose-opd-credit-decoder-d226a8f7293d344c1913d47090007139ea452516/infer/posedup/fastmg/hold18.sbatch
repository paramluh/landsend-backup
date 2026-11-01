#!/bin/bash
#SBATCH -J posedup-fastdev18 -N1 -n1 -c 16 --mem=200G --gres=gpu:4 -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/fastmg/logs/hold-%j.log
# dev allocation for the multi-GPU engine work (fastmg). slinky-18 GPUs 4-7 (0-3 = the backends agent).
# every test runs with CUDA_VISIBLE_DEVICES within 0-5 via srun --overlap --jobid <this>. Ends by scancel (my dev job) or at the time limit.
hostname; nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu --format=csv,noheader
sleep 43000
