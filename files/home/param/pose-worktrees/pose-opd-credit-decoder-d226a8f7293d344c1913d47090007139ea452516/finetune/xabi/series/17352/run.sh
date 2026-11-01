#!/bin/bash
# aloe-v2pure-17352 (10-22, user 12:22 PT via coordinator: both scaling FTs concurrently on 2 x 8 PPM hosts after the delivery-2 run):
#   copy of aloe-v2pure-16610/run.sh (job 26089: 8 nodes x 8 ranks = world 64, the same layout); changed: QQA_INIT + its assertion (PPM 8,1,1
#   step 17352) + an init COMPLETE check, dirs, MASTER_PORT 29864, the 8 hosts (chosen by aloe-v2pure-17641/tools/ftb_chain.sh from /scratch,
#   passed with sbatch -w; asserted: 8 nodes, none of slinky-4/10/15/18), ALOE_FINAL_SAVE from FTB_FINAL_SAVE (operational: 0 = no final
#   resumable save when /scratch is short; run.env [SPACE] already has no hourly save), and the /scratch check = this run's real per-host
#   footprint: final save 79 GB (if on) + final export 11 + 2 intermediate exports in flight 22 (tools/xfree_run.sh moves each to HF during
#   the run) + caches 2 + 30 GB left (+ 7 on rank 0's host).
# aloe-v2pure-ft (10-17) trainer launch, exec'd by aloe_v2pure.sbatch. = aloe-uncond-ft/aloe_uncond_run.sh (job 22317) with env aloe_v2pure.env:
# 22317's recipe (channel A empty, B = speech + pose, same LRs / weights / schedule shape / evals, init PPM main step 11488) on the Aloe pose
# release v2 (= the 8B SAM pose v2pure fine-tune's data). Differences in this script vs 22317's: no store staging (stores read in place from
# Weka), cpus-per-task explicit, FULL resumable saves (ALOE_SAVE_ROOT, node-local), RUNNING_JOB.txt, and an IN-JOB restart after a stop when
# control/RESTART_AFTER_STOP exists (the stop/resume test; the job keeps its nodes).
# STOP (save + clean exit): touch control/STOP_AFTER_SAVE   (or: scancel -b -s USR1 <job>)
# RESUME: sbatch aloe_v2pure.sbatch (ALOE_RESUME=auto continues from the newest complete node-local save; needs the same 3 nodes)
QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17352
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-v2pure-17352; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source ${QQA_ENV:-$Q/run.env}; set +a
export ALOE_FINAL_SAVE=${FTB_FINAL_SAVE:-1}
export QQA_INIT MASTER_PORT=29864
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$QQA_POSE_B" = 1 ] && [ "$QQA_A_EMPTY" = 1 ] && [ "$QQA_POSE_HEAD_BUILD" = 0 ] || { echo "ABORT: needs QQA_POSE_B=1 QQA_A_EMPTY=1 QQA_POSE_HEAD_BUILD=0"; exit 1; }
[ "$QQA_INIT" = /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17352 ] || { echo "ABORT: init must be PPM 8,1,1 step 17352 (user 10-22 scaling series)"; exit 1; }
[ -f $QQA_INIT/COMPLETE ] && [ "$(cat $QQA_INIT/COMPLETE)" = 17352 ] && [ -f $QQA_INIT/rep.pt ] && [ -f $QQA_INIT/meta.json ] || { echo "ABORT: init $QQA_INIT not COMPLETE"; exit 1; }
grep -q '@' ${QQA_ENV:-$Q/run.env} && { echo "ABORT: env still has @placeholders@"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe_pose.py $C/train_aloe_pose.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
[ "$SLURM_NNODES" = 8 ] || { echo "ABORT: needs 8 nodes (world 64 = 26089), got $SLURM_NNODES"; exit 1; }
case ",$PPM_HOSTS," in *,slinky-4,*|*,slinky-10,*|*,slinky-15,*|*,slinky-18,*) echo "ABORT: forbidden node in $PPM_HOSTS (slinky-10 rail 7 / spares)"; exit 1;; esac
export R0=$(echo $PPM_HOSTS | cut -d, -f1)
# one run tag for the whole run (first job's id), reused on every resume so metrics.jsonl stays one file: RUN_TAG.txt (delete it + the saves
# to start a fresh run)
if [ -f $Q/RUN_TAG.txt ]; then export QQA_RUN=$(cat $Q/RUN_TAG.txt); else export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID; echo $QQA_RUN > $Q/RUN_TAG.txt; fi
mkdir -p $QQA_CONTROL_DIR
echo "job $SLURM_JOB_ID nodes $PPM_HOSTS started $(TZ=America/Los_Angeles date '+%F %T PT') run $QQA_RUN" > $Q/RUNNING_JOB.txt
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-v2pure-17352 job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * SLURM_NNODES))) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|PPM_CD|PPM_DEC_ATTN|S3_MIN)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
# ---- node checks: /scratch need (header), our 8 GPUs idle + not thermally throttled, stores readable
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=2 --cpu-bind=none bash -c '
H=$(hostname -s); mkdir -p /scratch/amartyavarma/aloe-v2pure-17352
fr=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); used=$(du -sBG /scratch/amartyavarma/aloe-v2pure-17352 2>/dev/null | cut -f1 | tr -dc 0-9)
NEED=$(( 65 + 79 * ALOE_FINAL_SAVE )); [ "$H" = "$R0" ] && NEED=$((NEED + 7))
echo "$H /scratch $fr GB free (+ ${used:-0} GB already in this run dir), need $NEED GB (final save $ALOE_FINAL_SAVE)"; [ $((fr + ${used:-0})) -ge $NEED ] || { echo "$H ABORT: /scratch $fr GB free < $NEED GB"; exit 1; }
for g in 0 1 2 3 4 5 6 7; do q=$(nvidia-smi -i $g --query-gpu=index,memory.used,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader,nounits)
  [ "$(echo "$q" | cut -d, -f2 | tr -dc 0-9)" -lt 1000 ] || { echo "$H ABORT: GPU $g busy: $q"; exit 1; }
  echo "$q" | cut -d, -f3,4 | tr -d " " | grep -qE "(^|,)Active(,|$)" && { echo "$H ABORT: GPU $g thermal slowdown: $q"; exit 1; }; done
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
