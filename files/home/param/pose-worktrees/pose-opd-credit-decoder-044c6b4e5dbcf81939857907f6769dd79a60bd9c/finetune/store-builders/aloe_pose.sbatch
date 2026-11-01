#!/bin/bash
#SBATCH -J aloeposeft -p all -N8 --ntasks-per-node=8 --gres=gpu:8 -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-pose-ft/logs/aloeposeft-%j.log
# aloe-pose-ft trainer (code/train_aloe_pose.py + data_aloe_pose), world 64 on slinky-10,11,12,13,14,16,17,19.
# usage: QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 sbatch -w <8 nodes> --cpus-per-task=5 --mem=540G aloe.sbatch
# Stop = save-and-exit: touch aloe-ft/control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>): weights-only export at the next step boundary, then exit.
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-pose-ft; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
: "${QQA_INIT:?}"
set -a; source ${QQA_ENV:-$Q/aloe_pose.env}; set +a
export QQA_INIT MASTER_PORT=29841
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe_pose.py $C/train_aloe_pose.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-pose-ft job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * SLURM_NNODES))) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|PPM_CD|PPM_DEC_ATTN|S3_MIN)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_pose.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
