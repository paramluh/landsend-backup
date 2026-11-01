#!/bin/bash
# aloe-v2pure-17352 (10-22): copy of aloe-v2pure-16610/run.sh (job 26089, backup run.sh.from-26089); changed: QQA_INIT + its assertion (PPM 8,1,1
#   step 17352), dirs, MASTER_PORT 29864, and the LAUNCH LAYOUT: a 2-component het job (run.sbatch) = slinky-15 x8 + slinky-4 GPUs 0-5 x6 +
#   slinky-18 GPUs 2-7 x6 = world 20 (PPM_HOST_NTASKS / PPM_HOST_GPUS -> code/task_ppm.sh; 26089: 8 nodes x 8 = world 64; 23051 ran at world 24).
#   The v2pure plan spreads each step's same 64 rows over WORLD x R micros (maths-neutral). Node checks: /scratch per host = its rank count x
#   (final save 25.9 + 6 exports x 4.3 GB) + 25 GB (no hourly saves, run.env [SPACE]); our GPUs idle + not thermally throttled; init present.
# (26089's header) aloe-v2pure-ft (10-17) trainer launch. STOP (save + clean exit): touch control/STOP_AFTER_SAVE (or: scancel -b -s USR1 <job>)
# RESUME: sbatch run.sbatch (ALOE_RESUME=auto continues from the newest complete node-local save; needs the same nodes / world)
QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17352
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-v2pure-17352; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source ${QQA_ENV:-$Q/run.env}; set +a
export QQA_INIT MASTER_PORT=29864
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$QQA_POSE_B" = 1 ] && [ "$QQA_A_EMPTY" = 1 ] && [ "$QQA_POSE_HEAD_BUILD" = 0 ] || { echo "ABORT: needs QQA_POSE_B=1 QQA_A_EMPTY=1 QQA_POSE_HEAD_BUILD=0"; exit 1; }
[ "$QQA_INIT" = /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17352 ] || { echo "ABORT: init must be PPM 8,1,1 step 17352 (user 10-22 scaling series)"; exit 1; }
[ -f $QQA_INIT/COMPLETE ] && [ "$(cat $QQA_INIT/COMPLETE)" = 17352 ] && [ -f $QQA_INIT/rep.pt ] && [ -f $QQA_INIT/meta.json ] || { echo "ABORT: init $QQA_INIT not COMPLETE"; exit 1; }
grep -q '@' ${QQA_ENV:-$Q/run.env} && { echo "ABORT: env still has @placeholders@"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe_pose.py $C/train_aloe_pose.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
# ---- launch layout (het job): hosts sorted -V (rank 0 = slinky-4 GPU 0), per-host task counts + physical GPUs
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0,$SLURM_JOB_NODELIST_HET_GROUP_1" | sort -V | paste -sd,)
[ "$PPM_HOSTS" = slinky-4,slinky-15,slinky-18 ] || { echo "ABORT: hosts $PPM_HOSTS != slinky-4,slinky-15,slinky-18"; exit 1; }
export PPM_HOST_NTASKS=6,8,6 PPM_HOST_GPUS='slinky-4:0,1,2,3,4,5;slinky-15:0,1,2,3,4,5,6,7;slinky-18:2,3,4,5,6,7'
WORLD=20
# one run tag for the whole run (first job's id), reused on every resume so metrics.jsonl stays one file: RUN_TAG.txt (delete it + the saves
# to start a fresh run)
if [ -f $Q/RUN_TAG.txt ]; then export QQA_RUN=$(cat $Q/RUN_TAG.txt); else export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID; echo $QQA_RUN > $Q/RUN_TAG.txt; fi
mkdir -p $QQA_CONTROL_DIR
echo "job $SLURM_JOB_ID nodes $PPM_HOSTS started $(TZ=America/Los_Angeles date '+%F %T PT') run $QQA_RUN" > $Q/RUNNING_JOB.txt
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-v2pure-17352 job $SLURM_JOB_ID on $PPM_HOSTS (world $WORLD = $PPM_HOST_NTASKS; GPUs $PPM_HOST_GPUS) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|PPM_CD|PPM_DEC_ATTN|S3_MIN|PPM_HOST)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
# ---- node checks (one task per host): /scratch need, our GPUs idle (< 1 GB used) and not thermally throttled, stores readable
srun --het-group=0 -N1 -n1 --cpus-per-task=2 --cpu-bind=none : --het-group=1 -N2 -n2 --ntasks-per-node=1 --cpus-per-task=2 --cpu-bind=none bash -c '
H=$(hostname -s); mkdir -p /scratch/amartyavarma/aloe-v2pure-17352
G=$(echo "$PPM_HOST_GPUS" | tr ";" "\n" | grep "^$H:" | cut -d: -f2); LW=$(echo $G | tr , "\n" | wc -l); NEED=$((LW * 52 + 25))
fr=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); used=$(du -sBG /scratch/amartyavarma/aloe-v2pure-17352 2>/dev/null | cut -f1 | tr -dc 0-9)
echo "$H /scratch $fr GB free (+ ${used:-0} GB already in this run dir), need $NEED GB for $LW ranks; GPUs $G"
[ $((fr + ${used:-0})) -ge $NEED ] || { echo "$H ABORT: /scratch $fr GB free < $NEED GB"; exit 1; }
for g in $(echo $G | tr , " "); do
  q=$(nvidia-smi -i $g --query-gpu=index,memory.used,temperature.gpu,clocks.max.sm,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader,nounits)
  echo "$H gpu $q"; mu=$(echo "$q" | cut -d, -f2 | tr -dc 0-9)
  [ "$mu" -lt 1000 ] || { echo "$H ABORT: GPU $g busy ($mu MiB used)"; exit 1; }
  echo "$q" | cut -d, -f5,6 | tr -d " " | grep -qE "(^|,)Active(,|$)" && { echo "$H ABORT: GPU $g thermal slowdown active"; exit 1; }
done
ls /data/home/amartyavarma/ppm/aloe-v2pure-ft/stores/STORES.json > /dev/null || exit 1' || { echo "ABORT: node checks failed"; exit 1; }
rc=0
while :; do
  rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE $QQA_CONTROL_DIR/STOPPED
  srun --het-group=0 -N1 -n8 --ntasks-per-node=8 --cpus-per-task=${ALOE_CPT:-4} --kill-on-bad-exit=1 : \
       --het-group=1 -N2 -n12 --ntasks-per-node=6 --cpus-per-task=${ALOE_CPT:-4} --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_pose.py &
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
