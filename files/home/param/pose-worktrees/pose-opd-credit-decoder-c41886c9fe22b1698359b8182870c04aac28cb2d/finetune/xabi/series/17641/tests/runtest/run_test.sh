#!/bin/bash
# aloe-v2pure-17641 (10-22): copy of aloe-v2pure-16610/run.sh (job 26089, backup run.sh.from-26089); changed: QQA_INIT + its assertion (PPM 8,1,1
#   step 17641) + an init COMPLETE check, dirs, MASTER_PORT 29865, and the LAUNCH LAYOUT: world 64 (= 26089, so the same micros per rank) spread
#   over the 16 PPM hosts (whole nodes allocated) with per-host rank counts FTB_NTASKS (aligned with the hosts sorted -V; chosen by tools/ftb_chain.sh
#   from each host's /scratch), launched with srun --distribution=arbitrary; each host's ranks use the first of its Slurm-allocated GPU indices
#   (code/task_ppm.sh PPM_HOST_NTASKS / PPM_HOST_GPUS). Node checks: /scratch per host >= its ranks x 775 / WORLD GB (final save + final export +
#   2 intermediate exports in flight: tools/xfree_run.sh uploads them to HF and deletes them during the run) + 46 GB (node.pt 14 + caches + 30 left)
#   (+ 14 on rank 0's host); our GPUs idle and not thermally throttled; forbidden hosts slinky-4/10/15/18.
# STOP (save + clean exit): touch control/STOP_AFTER_SAVE (or: scancel -b -s USR1 <job>). RESUME: same sbatch command (same hosts, FTB_NTASKS).
QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17641
QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17352  # TEST
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-v2pure-17641/tests/runtest; C=/data/home/amartyavarma/ppm/aloe-v2pure-17641/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source /data/home/amartyavarma/ppm/aloe-v2pure-17641/run.env; set +a; QQA_CONTROL_DIR=$Q/control
export QQA_INIT MASTER_PORT=29865
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$QQA_POSE_B" = 1 ] && [ "$QQA_A_EMPTY" = 1 ] && [ "$QQA_POSE_HEAD_BUILD" = 0 ] || { echo "ABORT: needs QQA_POSE_B=1 QQA_A_EMPTY=1 QQA_POSE_HEAD_BUILD=0"; exit 1; }
[ "$QQA_INIT" = /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17352 ] || { echo "ABORT: init must be PPM 8,1,1 step 17641 (user 10-22 scaling series)"; exit 1; }
[ -f $QQA_INIT/COMPLETE ] && [ "$(cat $QQA_INIT/COMPLETE)" = 17352 ] && [ -f $QQA_INIT/rep.pt ] && [ -f $QQA_INIT/meta.json ] || { echo "ABORT: init $QQA_INIT not COMPLETE"; exit 1; }
grep -q '@' /data/home/amartyavarma/ppm/aloe-v2pure-17641/run.env && { echo "ABORT: env still has @placeholders@"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe_pose.py $C/train_aloe_pose.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
# ---- launch layout
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
case ",$PPM_HOSTS," in *,slinky-4,*|*,slinky-15,*|*,slinky-18,*) echo "ABORT: forbidden node in $PPM_HOSTS (slinky-10 rail 7 / spares)"; exit 1;; esac
: "${FTB_NTASKS:?per-host rank counts aligned with the hosts sorted -V}"
export PPM_HOST_NTASKS=$FTB_NTASKS
NN=$(echo $PPM_HOSTS | tr , '\n' | wc -l); WORLD=$(echo $PPM_HOST_NTASKS | tr , '\n' | awk '{s += $1} END {print s}')
[ "$(echo $PPM_HOST_NTASKS | tr , '\n' | wc -l)" = "$NN" ] || { echo "ABORT: FTB_NTASKS $FTB_NTASKS has not $NN entries"; exit 1; }
[ "$WORLD" = 5 ] || { echo "ABORT: world $WORLD != 64 (26089's)"; exit 1; }
# GPUs: this job's Slurm IDX per host (scontrol -d), first <count> of them
GPUMAP=$(scontrol show job -d $SLURM_JOB_ID | grep -oE "Nodes=[^ ]+ .*GRES=gpu[^ ]*IDX:[0-9,-]+" | while read -r line; do
  nl=$(echo "$line" | grep -oE "^Nodes=[^ ]+" | cut -d= -f2); ix=$(echo "$line" | grep -oE "IDX:[0-9,-]+" | cut -d: -f2)
  ex=$(echo $ix | tr , '\n' | while IFS=- read a b; do seq $a ${b:-$a}; done | paste -sd,)
  for h in $(scontrol show hostnames "$nl"); do echo "$h:$ex"; done; done)
PPM_HOST_GPUS=""
for h in $(echo $PPM_HOSTS | tr , ' '); do
  c=$(paste -d' ' <(echo $PPM_HOSTS | tr , '\n') <(echo $PPM_HOST_NTASKS | tr , '\n') | awk -v h=$h '$1==h{print $2}')
  g=$(echo "$GPUMAP" | grep "^$h:" | cut -d: -f2 | tr , '\n' | head -$c | paste -sd,)
  [ "$(echo $g | tr , '\n' | grep -c .)" = "$c" ] || { echo "ABORT: $h has $(echo $g | tr , '\n' | grep -c .) allocated GPUs < $c ranks (map: $(echo "$GPUMAP" | grep "^$h:"))"; exit 1; }
  PPM_HOST_GPUS="${PPM_HOST_GPUS:+$PPM_HOST_GPUS;}$h:$g"
done
export PPM_HOST_GPUS
HF=$Q/control/hostfile-$SLURM_JOB_ID
paste -d' ' <(echo $PPM_HOSTS | tr , '\n') <(echo $PPM_HOST_NTASKS | tr , '\n') | while read h c; do for i in $(seq 1 $c); do echo $h; done; done > $HF
# one run tag for the whole run (first job's id), reused on every resume so metrics.jsonl stays one file
if [ -f $Q/RUN_TAG.txt ]; then export QQA_RUN=$(cat $Q/RUN_TAG.txt); else export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID; echo $QQA_RUN > $Q/RUN_TAG.txt; fi
mkdir -p $QQA_CONTROL_DIR
echo "job $SLURM_JOB_ID nodes $PPM_HOSTS started $(TZ=America/Los_Angeles date '+%F %T PT') run $QQA_RUN" > $Q/RUNNING_JOB.txt
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-v2pure-17641 job $SLURM_JOB_ID on $PPM_HOSTS (world $WORLD = $PPM_HOST_NTASKS; GPUs $PPM_HOST_GPUS) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|PPM_CD|PPM_DEC_ATTN|S3_MIN|PPM_HOST)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
# ---- node checks (one task per host)
R0=$(echo $PPM_HOSTS | cut -d, -f1)
export R0 WORLD
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=2 --cpu-bind=none bash -c '
H=$(hostname -s); mkdir -p /scratch/amartyavarma/aloe-v2pure-17641
G=$(echo "$PPM_HOST_GPUS" | tr ";" "\n" | grep "^$H:" | cut -d: -f2); LW=$(echo $G | tr , "\n" | wc -l)
NEED=10; [ "$H" = "$R0" ] && NEED=$((NEED + 14))
fr=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); used=$(du -sBG /scratch/amartyavarma/aloe-v2pure-17641 2>/dev/null | cut -f1 | tr -dc 0-9)
echo "$H /scratch $fr GB free (+ ${used:-0} GB already in this run dir), need $NEED GB for $LW ranks; GPUs $G"
[ $((fr + ${used:-0})) -ge $NEED ] || { echo "$H ABORT: /scratch $fr GB free < $NEED GB"; exit 1; }
for g in $(echo $G | tr , " "); do
  q=$(nvidia-smi -i $g --query-gpu=index,memory.used,temperature.gpu,clocks.max.sm,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader,nounits)
  mu=$(echo "$q" | cut -d, -f2 | tr -dc 0-9)
  [ "$mu" -lt 1000 ] || { echo "$H ABORT: GPU $g busy ($mu MiB used)"; exit 1; }
  echo "$q" | cut -d, -f5,6 | tr -d " " | grep -qE "(^|,)Active(,|$)" && { echo "$H ABORT: GPU $g thermal slowdown active: $q"; exit 1; }
done
ls /data/home/amartyavarma/ppm/aloe-v2pure-ft/stores/STORES.json > /dev/null || exit 1; mkdir -p /scratch/amartyavarma/aloe-v2pure-17641; rmdir /scratch/amartyavarma/aloe-v2pure-17641 2>/dev/null; true' || { echo "ABORT: node checks failed"; exit 1; }
rc=0
while :; do
  rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE $QQA_CONTROL_DIR/STOPPED
  SLURM_HOSTFILE=$HF srun -n$WORLD --distribution=arbitrary --cpus-per-task=${ALOE_CPT:-4} --kill-on-bad-exit=1 bash $C/task_ppm.sh -c "import os,socket,torch; e=os.environ; print(socket.gethostname(), e['RANK'], e['LOCAL_RANK'], e['WORLD_SIZE'], e['LOCAL_WORLD_SIZE'], e['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count(), flush=True)" &
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
