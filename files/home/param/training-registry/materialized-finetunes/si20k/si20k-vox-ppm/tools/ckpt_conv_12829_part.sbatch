#!/bin/bash
#SBATCH -J si20k-ckconv -p all --ntasks-per-node=1 -c 8 --mem=64G --gres=none -t 1:00:00
#SBATCH -o /data/home/amartyavarma/ppm/si20k-vox-ppm/logs/ckconv-12829-part-%j.log
# PART variant (10-18 PT): converts the ranks of the hosts it is submitted on (-N <n> -w <hosts>); no finalize (separate job). Else
# = /data/home/amartyavarma/ppm/aloe-v2init-12529/ckpt_conv.sbatch for PPM's last training save dcp-12829 (job 22613 STOP_AFTER_SAVE, 10-17):
# node-local fp32 shards (world 128) on its 16 hosts -> ppm-local-bf16-shards-v1 inference copy on /data (~88.4 GB, as local-12529), the init
# format train_si20k_vox.load_init reads (meta.json + rank*.pt + rep.pt + COMPLETE). CPU only. Submit with -w <the 16 PPM hosts> (each task
# converts the ranks its own host holds; rank R = 8 x index of the host in the sorted list):
#   sbatch -w slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-17,slinky-19 \
#          /data/home/amartyavarma/ppm/si20k-vox-ppm/tools/ckpt_conv_12829.sbatch
# ckpt_local_export.py's floor check (free - 90 >= 300 GB) runs only in the rank-0 host's task and does not stop the other 15 tasks, so the
# same >= 390 GB check is done here once, before any task writes.
set -uo pipefail
SRC=${SRC:-/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-12829}
OUT=${OUT:-/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-12829}
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; C=/data/home/amartyavarma/ppm/aloe-vox-ft/code
echo "== ckconv $SLURM_JOB_ID $SRC -> $OUT on $(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,) $(TZ=America/Los_Angeles date '+%F %T PT')"
[ -f "$OUT/COMPLETE" ] && { echo "== $OUT already COMPLETE (step $(cat $OUT/COMPLETE)): nothing to do"; exit 0; }
FREE=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 390 ] || { echo "ABORT: /data free ${FREE} GB: the ~88 GB conversion would go below the 300 GB floor"; exit 1; }
echo "/data free ${FREE} GB before conversion"
srun --ntasks-per-node=1 nice -n 19 $PY $C/ckpt_local_export.py --src "$SRC" --out "$OUT" || exit 1
# finalize: separate job (ckpt_conv_12829_finalize), afterok every part (slinky-17 converts later: wfilter holds it)
echo "== ckconv done $(TZ=America/Los_Angeles date '+%T PT')  /data free $(df -BG --output=avail /data | tail -1 | tr -dc 0-9) GB  $(du -sh $OUT | cut -f1)"
