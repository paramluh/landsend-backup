#!/bin/bash
#SBATCH -J posedup-dl -w slinky-10 -N1 -n1 -c 4 --mem=24G --gres=none -t 02:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/logs/hfdl-%j.log
# pose-duplex export (job 30074) from our private HF -> slinky-10 node-local; every file size + LFS sha256 checked. usage: sbatch hfdl.sbatch <export tag>
source /data/home/amartyavarma/ppm/aloe-m1vox-live/env_live.sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
df -h /scratch | tail -1
HFW=32 HASHW=4 $PY -I /data/home/amartyavarma/ppm/aloe-posedup-live/tools/hfdl.py InternalCan/ppm-aloe-poseduplex-v5 $1 /scratch/amartyavarma/aloe-posedup-live
echo "rc=$? $(date)"; df -h /scratch | tail -1
