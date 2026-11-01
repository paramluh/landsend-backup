#!/bin/bash
#SBATCH -J ppm-qqa-ft -p all -N3 --ntasks-per-node=8 --gres=gpu:8 -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/qqa-ft/logs/qqa-%j.log
# qqa-ft trainer (code/train_qqa.py), world 24 on slinky-7/15/18. usage: QQA_INIT=<ppm local copy> sbatch -w slinky-7,slinky-15,slinky-18 --cpus-per-task=5 --mem=540G qqa.sbatch
# Stop = save-and-exit: touch qqa-ft/control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>): weights-only export at the next step boundary, then exit.
set -uo pipefail
Q=/data/home/amartyavarma/ppm/qqa-ft; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
: "${QQA_INIT:?}"
set -a; source ${QQA_ENV:-$Q/qqa.env}; set +a
export QQA_INIT
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/train_qqa.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== qqa-ft job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * SLURM_NNODES))) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|PPM_CD|PPM_DEC_ATTN|S3_MIN)' | sort
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_qqa.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
