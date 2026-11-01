#!/bin/bash
# copy_init.sh STEP : node-local dcp-STEP on the live PPM job's nodes -> ckpt-infer/ppm-v1/local-STEP (bf16 model weights only).
# Read-only, CPU-only overlap step, nice 19 + ionice idle (= the 2-side sample agent's path, samples/2side-local-10386/work/export.sh).
# Never signals / modifies the training job or its files. Logs PPM step time before / after.
set -uo pipefail
C=/data/home/amartyavarma/ppm/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
STEP=$1; JOB=$(squeue -u "$USER" -h -n ppm -t R -o %i | head -1); L=/data/home/amartyavarma/ppm/logs/ppm-$JOB.log
OUT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-$STEP
st() { grep -o "step [0-9]*/[0-9]* [0-9.]*s" $L | tail -${1:-10} | awk '{s+=$3} END {printf "%.2f s/step (n=%d)\n", s/NR, NR}'; }
echo "start $(date) job $JOB  PPM before: $(st 20)"
srun --jobid $JOB --overlap -N16 --ntasks-per-node=1 -c4 --cpu-bind=none --mem=24G -J ppm-ckexport nice -n 19 ionice -c3 \
  $PY $C/ckpt_local_export.py --src /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-$STEP --out $OUT || exit 1
srun --jobid $JOB --overlap -N1 -n1 -c1 --cpu-bind=none --mem=4G -J ppm-ckexport $PY $C/ckpt_local_export.py --out $OUT --finalize || exit 1
echo "end $(date)  PPM during/after (last 6 steps): $(st 6)"
