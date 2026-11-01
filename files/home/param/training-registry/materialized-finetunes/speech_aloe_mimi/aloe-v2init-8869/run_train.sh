#!/bin/bash
# aloe-v2init-8869 training block = aloe-v2-ft/run_train.sh (job 22383) with ONLY the init + run dirs changed (+ a launcher-side final-export
# guard: if /data would drop below 300 GB with the ~77 GB consolidated export, an empty placeholder at QQA_EXPORT makes the trainer refuse
# the /data export immediately (node-local final export is always written) instead of holding the nodes in its 120-min space wait).
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-v2init-8869; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-restore/ppm-v1-w-8869
set -a; source ${QQA_ENV:-$Q/aloe_v2init.env}; set +a
export QQA_INIT MASTER_PORT=29861
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ -f $QQA_INIT/COMPLETE ] || { echo "ABORT: init not COMPLETE"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe_v2.py $C/train_aloe_v2.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-v2init-8869 job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * SLURM_NNODES))) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|PPM_CD|PPM_DEC_ATTN|S3_MIN)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
MJ=$QQA_LOG_ROOT/$QQA_RUN/metrics.jsonl
( while sleep 15; do
    st=$(tail -n 1 $MJ 2>/dev/null | grep -o '"step": [0-9]*' | grep -o '[0-9]*$'); [ -n "$st" ] && [ "$st" -ge 245 ] || continue
    [ -e $QQA_EXPORT ] && continue
    fr=$(df -B1 --output=avail /data | tail -1); if [ $((fr / 1000000000 - 80)) -lt $QQA_EXPORT_MIN_FREE_GB ]; then
      mkdir -p $QQA_EXPORT && echo "$(date) /data free $((fr / 1000000000)) GB" > $QQA_EXPORT/PLACEHOLDER_NODE_SCRATCH_ONLY
      echo "== export guard $(TZ=America/Los_Angeles date '+%T PT'): /data free $((fr / 1000000000)) GB < 300 + 80 -> final export node-local only"; fi
  done ) &
GUARD=$!
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_v2.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
kill $GUARD 2>/dev/null
if [ -f $QQA_EXPORT/PLACEHOLDER_NODE_SCRATCH_ONLY ] && [ ! -f $QQA_EXPORT/COMPLETE ]; then rm -f $QQA_EXPORT/PLACEHOLDER_NODE_SCRATCH_ONLY; rmdir $QQA_EXPORT; echo "== removed export placeholder"; fi
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
