#!/bin/bash
#SBATCH -J genweb-avhybrid
#SBATCH -x slinky-9
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/%x-%j.out
# 10-26 videort: the PLAYGROUND's own avatar stream backends (one hybrid videort backend per GPU, registered with instance 1's avatar front;
# they take playground feeds first and render clips when idle), so a playground stream can never take the live call's avatar backend.
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) $(date)"
exec bash /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/hybrid_seq.sh $CUDA_VISIBLE_DEVICES   # 10-26: seq sampler, one 2-GPU backend (was start_hybrid.sh)
