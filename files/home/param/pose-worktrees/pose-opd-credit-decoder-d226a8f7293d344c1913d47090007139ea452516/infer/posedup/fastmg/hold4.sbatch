#!/bin/bash
#SBATCH -J posedup-fastdev -N1 -n1 -c 32 --mem=400G --gres=gpu:8 -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/fastmg/logs/hold-%j.log
# dev allocation for the multi-GPU engine work (fastmg). All 8 GPUs held so the throttled 6-7 are never handed to anyone else's job;
# every test runs with CUDA_VISIBLE_DEVICES within 0-5 via srun --overlap --jobid <this>. Ends by scancel (my dev job) or at the time limit.
hostname; nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu --format=csv,noheader
sleep 43000
