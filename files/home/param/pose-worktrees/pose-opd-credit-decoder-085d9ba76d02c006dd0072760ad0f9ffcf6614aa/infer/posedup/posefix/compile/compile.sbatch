#!/bin/bash
#SBATCH -J posefix-compile
#SBATCH -w slinky-4
#SBATCH -x slinky-9
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 2:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/compile/compile-%j.log
D=/data/home/amartyavarma/ppm/aloe-posedup-live
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,memory.used --format=csv,noheader
echo "CVD=$CUDA_VISIBLE_DEVICES"
cd /tmp
bash $D/posefix/compile/run_dup.sh $SLURM_JOB_ID > $D/posefix/compile/dup-$SLURM_JOB_ID.log 2>&1
echo "dup exit $? $(date -u +%T)"
bash $D/posefix/compile/run_mono.sh $SLURM_JOB_ID > $D/posefix/compile/mono-$SLURM_JOB_ID.log 2>&1
echo "mono exit $? $(date -u +%T)"
