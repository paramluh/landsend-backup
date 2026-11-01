#!/bin/bash
#SBATCH -J ppm-ckexport16610 -p all -N16 -w slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-17,slinky-19 --ntasks-per-node=1 --cpus-per-task=8 --mem=48G -t 01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/samdata-ft/logs/export16610-%j.log
# PPM pretraining's final node-local save dcp-16610 (job 25627, stopped by the user 10-20 16:22 PT) -> inference checkpoint
# /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-16610 (ppm-local-bf16-shards-v1 = the format every PPM Aloe FT inits from, e.g. local-11488),
# with the existing tool ppm/code/ckpt_local_export.py (read-only on the save; CPU only; one task per node that holds ranks). The dcp is NOT touched.
set -uo pipefail
C=/data/home/amartyavarma/ppm/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
SRC=/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-16610; OUT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-16610
echo "== export $SRC -> $OUT on $SLURM_JOB_NODELIST $(TZ=America/Los_Angeles date '+%F %T PT')"
srun -N16 --ntasks-per-node=1 -c8 --cpu-bind=none --mem=44G nice -n 10 $PY $C/ckpt_local_export.py --src $SRC --out $OUT || { echo "EXPORT FAILED"; exit 1; }
srun -N1 -n1 -c1 --cpu-bind=none --mem=4G $PY $C/ckpt_local_export.py --out $OUT --finalize || exit 1
echo "== done $(TZ=America/Los_Angeles date '+%T PT')"; ls $OUT | wc -l; du -sh $OUT
