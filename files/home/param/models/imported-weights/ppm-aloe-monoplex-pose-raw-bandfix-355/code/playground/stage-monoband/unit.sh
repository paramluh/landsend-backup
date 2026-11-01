#!/bin/bash
# 10-30 monohost subagent (main -a8; user 00:10 PT "instahost the monoplex when it's done"): hot worker for weights id monoband355
# (pose6 monoband export-355) = genweb/mono_unit.sh with CUDA_VISIBLE_DEVICES pinned to this job's own GPU IDX (spare GPUs are not isolated).
IDX=$(scontrol show job -d $SLURM_JOB_ID | grep -o 'IDX:[0-9,-]*' | head -1 | cut -d: -f2)
case "$IDX" in *-*) a=${IDX%-*}; b=${IDX#*-}; IDX=$(seq -s, $a $b);; esac
export CUDA_VISIBLE_DEVICES=$IDX
echo "monoband unit $SLURM_JOB_ID on $(hostname) IDX=$IDX CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i $CUDA_VISIBLE_DEVICES
exec bash /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/mono_unit.sh
