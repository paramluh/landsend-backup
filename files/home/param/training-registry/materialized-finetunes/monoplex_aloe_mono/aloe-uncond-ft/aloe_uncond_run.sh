#!/bin/bash
# aloe-uncond-ft (10-16) trainer launch, exec'd by aloe_uncond.sbatch (job gate: afterany 22291 on its 8 nodes, starts at control/READY).
# = aloe-nopose-ft/aloe_nopose.sbatch (22291 = 22091's sbatch + store staging) with env aloe_uncond.env: UNCONDITIONAL channel B = 22091
# (QQA_POSE_B=1, pose maths unchanged) with channel A EMPTY (QQA_A_EMPTY=1), word-lane loss x10 (QQA_LW_WORD=10, speech / pose x1), pose head
# not built (QQA_POSE_HEAD_BUILD=0). World 64 on slinky-1,10,11,13,14,16,17,19; every node holds ALL 8 stores (as 22091 / 22291), so the
# rank -> row map is unchanged. Init = 22091's (PPM step 11488). Stop = save-and-exit: touch aloe-uncond-ft/control/STOP_AFTER_SAVE (or
# scancel -b -s USR1 <job>).
QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-uncond-ft; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
: "${QQA_INIT:?}"
set -a; source ${QQA_ENV:-$Q/aloe_uncond.env}; set +a
export QQA_INIT MASTER_PORT=29845
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$QQA_POSE_B" = 1 ] && [ "$QQA_A_EMPTY" = 1 ] && [ "$QQA_POSE_HEAD_BUILD" = 0 ] || { echo "ABORT: uncond run needs QQA_POSE_B=1 QQA_A_EMPTY=1 QQA_POSE_HEAD_BUILD=0"; exit 1; }
[ "$QQA_INIT" = /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 ] || { echo "ABORT: init must be 22091's (PPM step 11488)"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe_pose.py $C/train_aloe_pose.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-uncond-ft job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * SLURM_NNODES))) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|PPM_CD|PPM_DEC_ATTN|S3_MIN)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
# ---- stores: verify on every node, stage from Weka where missing / different
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=5 --cpu-bind=none bash -c '
S=/data/home/amartyavarma/ppm/aloe-pose-ft/stores; D=/scratch/amartyavarma/aloe-pose-ft/stores; H=$(hostname -s); PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
chk() { S3_MIN_FRAMES=125 PPM_CD=2 $PY -I -c "
import sys, json; sys.path.insert(0, \"/data/home/amartyavarma/ppm/aloe-uncond-ft/code/lib\"); import s3rows as F
ref = json.load(open(\"$S/STORES.json\")); bad = []
for k in range(8):
    try:
        ok = F.load_node(\"$D/p%d\" % k)[\"digest\"] == ref[\"p%d\" % k][\"node_digest\"]
    except Exception:
        ok = False
    bad += [] if ok else [str(k)]
print(\" \".join(bad))"; for k in 0 1 2 3 4 5 6 7; do
  [ "$(find $S/p$k -type f -printf "%P %s\n" 2>/dev/null | sort | md5sum)" = "$(find $D/p$k -type f -printf "%P %s\n" 2>/dev/null | sort | md5sum)" ] || echo -n " $k"; done; }
mkdir -p $D; rm -rf $D/p?.tmp; bad=$(chk | tr " " "\n" | sort -u | paste -sd" " | sed "s/^ *//")
if [ -n "$bad" ]; then
  echo "$H staging stores [$bad] from Weka; $(df -h /scratch | tail -1)"
  for k in $bad; do ( rm -rf $D/p$k.tmp; cp -r $S/p$k $D/p$k.tmp && rm -rf $D/p$k && mv $D/p$k.tmp $D/p$k ) & done; wait
  for k in $bad; do
    a=$(find $S/p$k -type f -printf "%P %s\n" | sort | md5sum); b=$(find $D/p$k -type f -printf "%P %s\n" | sort | md5sum)
    [ "$a" = "$b" ] || { echo "$H p$k MISMATCH"; exit 1; }
  done
  bad=$(chk | tr " " "\n" | sort -u | paste -sd" " | sed "s/^ *//"); [ -z "$bad" ] || { echo "$H stores still bad: $bad"; exit 1; }
fi
echo "$H all 8 stores verified (digests == STORES.json + every file name / size == the Weka build); $(df -h /scratch | tail -1)"
fr=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); [ "$fr" -ge 100 ] || { echo "$H ABORT: /scratch $fr GB free < 100 GB"; exit 1; }; echo "$H /scratch $fr GB free (>= 100 GB)"' || { echo "ABORT: store staging failed"; exit 1; }
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_pose.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
