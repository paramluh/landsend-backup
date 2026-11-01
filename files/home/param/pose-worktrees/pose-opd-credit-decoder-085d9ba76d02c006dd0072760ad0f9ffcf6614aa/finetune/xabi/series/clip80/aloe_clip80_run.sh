#!/bin/bash
# aloe-v2pure-clip80 (10-18) trainer launch, exec'd by aloe_clip80.sbatch. = aloe-v2pure-ft/aloe_v2pure_run.sh (job 23051) with env
# aloe_clip80.env: 23051's recipe with ONE change, the system-prompt rule (no persona text; voice clip on a seeded 80% of items). Differences in
# this script vs 23051's: this run's dir / env / scratch dir / port; the /scratch need scales with the node count (3 nodes = world 24 as 23051,
# or 8 nodes = world 64: the plan keeps 64 rows per step at any world); the prompt-rule env is asserted.
# STOP (save + clean exit): touch control/STOP_AFTER_SAVE   (or: scancel -b -s USR1 <job>)
# RESUME: sbatch aloe_clip80.sbatch (ALOE_RESUME=auto continues from the newest complete node-local save; needs the same nodes / world)
QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-v2pure-clip80; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source ${QQA_ENV:-$Q/aloe_clip80.env}; set +a
export QQA_INIT MASTER_PORT=29857
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$QQA_POSE_B" = 1 ] && [ "$QQA_A_EMPTY" = 1 ] && [ "$QQA_POSE_HEAD_BUILD" = 0 ] || { echo "ABORT: needs QQA_POSE_B=1 QQA_A_EMPTY=1 QQA_POSE_HEAD_BUILD=0"; exit 1; }
[ "$QQA_INIT" = /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 ] || { echo "ABORT: init must be 23051's (PPM step 11488)"; exit 1; }
[ "$ALOE_PERSONA_SIDES" = none ] && [ "$ALOE_VOICE_RULE" = p10 ] && [ "$ALOE_VOICE_P10" = 8 ] && [ "$ALOE_VOICE_TAG" = aloe-v2pure-clip80 ] || { echo "ABORT: prompt rule env"; exit 1; }
grep -q '@' ${QQA_ENV:-$Q/aloe_clip80.env} && { echo "ABORT: env still has @placeholders@"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe_pose.py $C/train_aloe_pose.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
case ",$PPM_HOSTS," in *,slinky-4,*|*,slinky-7,*|*,slinky-15,*|*,slinky-18,*|*,slinky-9,*|*,slinky-1[0-4],*|*,slinky-16,*|*,slinky-19,*) echo "ABORT: forbidden node in $PPM_HOSTS"; exit 1;; esac
# one run tag for the whole run (first job's id), reused on every resume so metrics.jsonl stays one file: RUN_TAG.txt (delete it + the saves
# to start a fresh run)
if [ -f $Q/RUN_TAG.txt ]; then export QQA_RUN=$(cat $Q/RUN_TAG.txt); else export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID; echo $QQA_RUN > $Q/RUN_TAG.txt; fi
mkdir -p $QQA_CONTROL_DIR
echo "job $SLURM_JOB_ID nodes $PPM_HOSTS started $(TZ=America/Los_Angeles date '+%F %T PT') run $QQA_RUN" > $Q/RUNNING_JOB.txt
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-v2pure-clip80 job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * SLURM_NNODES))) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|PPM_CD|PPM_DEC_ATTN|S3_MIN)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
# ---- node checks: /scratch space for 2 full saves + 6 node-local exports (23051 at 3 nodes: ~540 GB per node; scales ~1/nodes:
#      need = 1500 / nodes + 50 GB), GPU clocks (no throttled GPU), stores readable
NEED=$((1500 / SLURM_NNODES + 50))
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=2 --cpu-bind=none bash -c '
H=$(hostname -s); mkdir -p /scratch/amartyavarma/aloe-v2pure-clip80
fr=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); used=$(du -sBG /scratch/amartyavarma/aloe-v2pure-clip80 2>/dev/null | cut -f1 | tr -dc 0-9)
echo "$H /scratch $fr GB free (+ ${used:-0} GB already in this run dir), need '$NEED' GB"; [ $((fr + ${used:-0})) -ge '$NEED' ] || { echo "$H ABORT: /scratch $fr GB free < '$NEED' GB"; exit 1; }
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,clocks_throttle_reasons.hw_thermal_slowdown --format=csv,noheader | sed "s/^/$H gpu /"
ls /data/home/amartyavarma/ppm/aloe-v2pure-ft/stores/STORES.json > /dev/null || exit 1' || { echo "ABORT: node checks failed"; exit 1; }
rc=0
while :; do
  rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE $QQA_CONTROL_DIR/STOPPED
  srun --ntasks-per-node=8 --cpus-per-task=${ALOE_CPT:-4} --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_pose.py &
  SRUN=$!
  while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
  echo "== trainer exit $rc $(TZ=America/Los_Angeles date '+%T PT')"
  if [ $rc = 0 ] && [ -f $QQA_CONTROL_DIR/STOPPED ] && [ -f $QQA_CONTROL_DIR/RESTART_AFTER_STOP ]; then
    rm -f $QQA_CONTROL_DIR/RESTART_AFTER_STOP
    echo "== stopped at step $(cat $QQA_CONTROL_DIR/STOPPED) with RESTART_AFTER_STOP: in-job resume (ALOE_RESUME=$ALOE_RESUME) $(TZ=America/Los_Angeles date '+%T PT')"
    continue
  fi
  break
done
echo "job $SLURM_JOB_ID nodes $PPM_HOSTS ended $(TZ=America/Los_Angeles date '+%F %T PT') rc $rc" >> $Q/RUNNING_JOB.txt
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
