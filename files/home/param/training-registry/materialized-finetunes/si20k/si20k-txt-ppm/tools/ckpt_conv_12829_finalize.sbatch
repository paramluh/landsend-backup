#!/bin/bash
#SBATCH -J si20ktxt-ckfin -p all -N1 -n1 -c1 --mem=2G --gres=none -t 0:20:00
#SBATCH -o /data/home/amartyavarma/ppm/si20k-txt-ppm/logs/ckconv-12829-finalize-%j.log
# finalize local-12829 (every rank file + rep.pt present -> COMPLETE) after all ckpt_conv_12829_part jobs (afterok).
OUT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-12829
/data/home/amartyavarma/venvs/qwen-s2/bin/python /data/home/amartyavarma/ppm/aloe-vox-ft/code/ckpt_local_export.py --out "$OUT" --finalize || exit 1
echo "== $(TZ=America/Los_Angeles date '+%T PT') local-12829 COMPLETE $(du -sh $OUT | cut -f1); /data free $(df -BG --output=avail /data | tail -1 | tr -dc 0-9) GB"
