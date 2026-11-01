#!/bin/bash
#SBATCH -J ppm -p all --ntasks-per-node=8 --cpus-per-task=8 --gres=gpu:8 --mem=560G -t 3-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/logs/ppm-%j.log
#SBATCH --signal=B:USR1@300
# PPM multi-node trainer. usage: PPM_ENV=<env file> sbatch -N <n> -w <hosts> ppm.sbatch      (env file: prod_ppm.env = every knob explicit)
# Rank = 8 x (index of this host in the SORTED allocation) + local id; MASTER = the first host. bf16 only.
set -uo pipefail
C=/data/home/amartyavarma/ppm/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
[ -n "${PPM_ENV:-}" ] && [ -f "$PPM_ENV" ] || { echo "ABORT: PPM_ENV not set / missing"; exit 1; }
PRE=$(env | grep '^PPM_' | grep -v '^PPM_ENV=')          # PPM_* given at submission override the env file (logged)
set -a; source "$PPM_ENV"; set +a
if [ -n "$PRE" ]; then while IFS= read -r l; do export "$l"; done <<< "$PRE"; echo "== submission overrides: $(echo $PRE | tr '\n' ' ')"; fi
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/train_ppm.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
export PPM_RUN_BASE=${PPM_RUN_BASE:-${PPM_RUN:-ppm}}             # checkpoints + control files per run base (stable across resumes)
export PPM_RUN=$PPM_RUN_BASE-j$SLURM_JOB_ID
export PPM_CONTROL_DIR=${PPM_CONTROL_DIR:-/data/home/amartyavarma/ppm/control/$PPM_RUN_BASE}; mkdir -p "$PPM_CONTROL_DIR"
# Slurm --signal=B:USR1@300 (and `scancel -b -s USR1 <job>`) reach THIS shell: save at the next step boundary, then exit cleanly
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch $PPM_CONTROL_DIR/STOP_AFTER_SAVE"; touch "$PPM_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== ppm job $SLURM_JOB_ID on $PPM_HOSTS ($SLURM_NNODES nodes, world $((8 * SLURM_NNODES))) $(TZ=America/Los_Angeles date '+%F %T PT') env $PPM_ENV run $PPM_RUN"
srun --mem=${PPM_MEM:-560G} --ntasks-per-node=8 --cpus-per-task=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_ppm.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
