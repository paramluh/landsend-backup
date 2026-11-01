#!/bin/bash
#SBATCH -J posedup-dl498 -N1 -n1 -c 16 --mem=32G --gres=none -t 01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/fastmg/logs/hfdl498-%j.log
# export-498 consolidated bf16 DCP from our PRIVATE HF repo -> this node's /scratch (same path as on slinky-10); size + LFS sha256 of every
# file checked by tools/hfdl.py (DL_VERIFY.json). usage: sbatch -w <host> -x slinky-9 hfdl498.sbatch
source /data/home/amartyavarma/ppm/aloe-m1vox-live/env_live.sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
hostname; df -h /scratch | tail -1
D=/scratch/amartyavarma/aloe-posedup-live
[ -f $D/export-498-dcp/COMPLETE ] && [ -f $D/export-498-dcp/DL_VERIFY.json ] && { echo "already staged"; cat $D/export-498-dcp/DL_VERIFY.json; exit 0; }
mkdir -p $D
HFW=32 HASHW=12 $PY -I /data/home/amartyavarma/ppm/aloe-posedup-live/tools/hfdl.py InternalCan/ppm-aloe-poseduplex-v5-final export-498-dcp $D
echo "rc=$? $(date)"; df -h /scratch | tail -1
