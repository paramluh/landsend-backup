#!/bin/bash
#SBATCH -J ppm811-hold -p all -w slinky-18 -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:2 --mem=240G -t 0-04:00:00
#SBATCH -o /data/home/amartyavarma/ppm/logs/ppm811test/hold-%j.log
# holds 2 free slinky-18 GPUs for the 8,1,1 reduced-depth correctness / timing tests; work runs as `srun --jobid <this> --overlap` steps. scancel when done.
echo "== hold $SLURM_JOB_ID on $(hostname -s) GPUs $CUDA_VISIBLE_DEVICES $(TZ=America/Los_Angeles date '+%F %T PT')"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,clocks.sm,clocks.max.sm,temperature.gpu --format=csv
df -h /scratch | tail -1
sleep infinity
