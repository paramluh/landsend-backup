#!/bin/bash
#SBATCH -J v2i-ckconv -p all -N16 --ntasks-per-node=1 -c 8 --mem=64G --gres=none -t 1:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2init-12529/logs/ckconv-%j.log
# PPM's node-local stop save dcp-N (fp32 shards of world 128 on its 16 hosts) -> ppm-local-bf16-shards-v1 inference copy on /data (~84 GB;
# ckpt_local_export.py asserts /data stays >= 300 GB free), = how the pose FT's init local-11488 was made. Submitted by launch.sh with
# -w <the 16 PPM hosts> and SRC / OUT exported; the training job depends on it (afterok).
set -uo pipefail
: "${SRC:?}" "${OUT:?}"
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; C=/data/home/amartyavarma/ppm/aloe-vox-ft/code
echo "== ckconv $SLURM_JOB_ID $SRC -> $OUT on $(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,) $(TZ=America/Los_Angeles date '+%F %T PT')"
srun --ntasks-per-node=1 nice -n 19 $PY $C/ckpt_local_export.py --src "$SRC" --out "$OUT" || exit 1
srun -N1 -n1 $PY $C/ckpt_local_export.py --out "$OUT" --finalize || exit 1
echo "== ckconv done $(TZ=America/Los_Angeles date '+%T PT')"
