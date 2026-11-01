#!/bin/bash
# aloe-next-6242 training block = aloe-v2init-8869/run_train.sh (= aloe-v2-ft/run_train.sh, job 22383) with ONLY: init (the 14B next-arch w-6242
# bf16 DCP export), run dirs / port, env file aloe_next.env (recipe env + the next.env arch block), trainer train_aloe_next.py (+ model_ilp /
# rope_frame compile checks). Launcher-side final-export guard as the recipe.
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-next-6242; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
export QQA_INIT=/data/home/amartyavarma/ppm/next-arch/ckpt-weka/ppm-next/w-6242
set -a; source ${QQA_ENV:-$Q/aloe_next.env}; set +a
export QQA_INIT MASTER_PORT=29871
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ -f $QQA_INIT/COMPLETE ] || { echo "ABORT: init not COMPLETE"; exit 1; }
for f in $C/model_ppm.py $C/model_ilp.py $C/rope_frame.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe_v2.py $C/train_aloe_next.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-next-6242 job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * SLURM_NNODES))) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|PPM_|ILP_|S3_MIN)' | sort
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
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_next.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
kill $GUARD 2>/dev/null
if [ -f $QQA_EXPORT/PLACEHOLDER_NODE_SCRATCH_ONLY ] && [ ! -f $QQA_EXPORT/COMPLETE ]; then rm -f $QQA_EXPORT/PLACEHOLDER_NODE_SCRATCH_ONLY; rmdir $QQA_EXPORT; echo "== removed export placeholder"; fi
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
