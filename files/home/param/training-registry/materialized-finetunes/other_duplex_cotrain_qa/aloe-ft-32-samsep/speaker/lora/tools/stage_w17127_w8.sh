#!/bin/bash
# OPTIONAL (10-24, NOT RUN): stage 29345's INIT w-17127 (PPMVOX v3 stop export, world 128, private HF InternalCan/ppmvox-v3-16861, sha256 == HF LFS)
# on slinky-18 for a world-8 identity eval that should reproduce 29345's step-0 eval 1.8484 (run_identity_eval.env with QQA_INIT / VOX_ADAPTER =
# /scratch/amartyavarma/aloespk-init-w8/w-17127). Needs ~78 GB: refuses unless /scratch keeps >= 30 GB after (today ~78 GB free: does NOT fit
# next to the staged export-249). usage: srun -p all -w slinky-18 -N1 -n1 -c32 --mem=64G -t 60 bash stage_w17127_w8.sh
set -euo pipefail
cd /tmp
F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
[ "$F" -ge $((78 + 30)) ] || { echo "ABORT: /scratch free ${F} GB < 78 + 30 floor"; exit 1; }
mkdir -p /scratch/amartyavarma/aloespk-init-w8
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 EVALVOX_HF_REPO=InternalCan/ppmvox-v3-16861
nice -n 5 /data/home/amartyavarma/venvs/duplex/bin/python -I /data/home/amartyavarma/ppm/aloe-ft-32-samsep/evalvox/stage_export.py w-17127 \
  /data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/tools/subset_w17127_w8.json 0 /scratch/amartyavarma/aloespk-init-w8/w-17127
