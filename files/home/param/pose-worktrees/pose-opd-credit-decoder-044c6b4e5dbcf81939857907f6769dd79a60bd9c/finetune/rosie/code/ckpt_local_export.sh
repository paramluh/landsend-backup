#!/bin/bash
# Node-local PPM training save -> inference checkpoint on /data, without touching the training process (read-only, CPU-only overlap step
# inside the running training job, one task per node, nice 19, ~90 s, ~84 GB written).
#   bash ckpt_local_export.sh [STEP]        STEP default = the newest dcp-N COMPLETE on the run's nodes (from the live log)
# env: PPM_RUN_BASE (ppm-v1), PPM_TRAIN_JOB (default: the running job named 'ppm'), PPM_INFER_CKPT_ROOT (/data/home/amartyavarma/ppm/ckpt-infer)
# -> $PPM_INFER_CKPT_ROOT/$PPM_RUN_BASE/local-STEP  (use as infer_ppm.sh --ckpt). Delete it once the trainer's own export w-N supersedes it.
set -uo pipefail
C=/data/home/amartyavarma/ppm/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
RB=${PPM_RUN_BASE:-ppm-v1}; ROOT=${PPM_INFER_CKPT_ROOT:-/data/home/amartyavarma/ppm/ckpt-infer}
JOB=${PPM_TRAIN_JOB:-$(squeue -u "$USER" -h -n ppm -t R -o %i | head -1)}
[ -n "$JOB" ] || { echo "no running training job (set PPM_TRAIN_JOB)"; exit 1; }
NN=$(squeue -j "$JOB" -h -o %D)
STEP=${1:-$(grep -ho "ckpt/$RB/dcp-[0-9]* node-local" /data/home/amartyavarma/ppm/logs/ppm-$JOB.log | tail -1 | grep -o "dcp-[0-9]*" | cut -d- -f2)}
[ -n "$STEP" ] || { echo "no node-local save found in ppm-$JOB.log (pass STEP)"; exit 1; }
OUT=$ROOT/$RB/local-$STEP
echo "job $JOB ($NN nodes): /scratch/amartyavarma/ppm/ckpt/$RB/dcp-$STEP -> $OUT"
srun --jobid "$JOB" --overlap -N"$NN" --ntasks-per-node=1 -c4 --cpu-bind=none --mem=24G -J ppm-ckexport nice -n 19 \
  $PY $C/ckpt_local_export.py --src /scratch/amartyavarma/ppm/ckpt/$RB/dcp-$STEP --out "$OUT" || exit 1
srun --jobid "$JOB" --overlap -N1 -n1 -c1 --cpu-bind=none --mem=4G -J ppm-ckexport $PY $C/ckpt_local_export.py --out "$OUT" --finalize
