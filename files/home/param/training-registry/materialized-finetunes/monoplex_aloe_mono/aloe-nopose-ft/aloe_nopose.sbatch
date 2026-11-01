#!/bin/bash
#SBATCH -J aloenoposeft -p all -N8 --ntasks-per-node=8 --gres=gpu:8 -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-nopose-ft/logs/aloenoposeft-%j.log
# aloe-nopose-ft (10-16): the NO-POSE ABLATION of job 22091 = aloe-pose-ft/aloe_pose.sbatch with env aloe_nopose.env (QQA_POSE_B=0) and code
# = 22091's code + the QQA_POSE_B flag (code/train_aloe_pose.py + data_aloe_pose.py). World 64 on slinky-1,10,11,13,14,16,17,19 (slinky-12 of
# 22091 is busy with Voxtral 22275 -> slinky-1). Every node holds ALL 8 stores (as 22091), so the rank -> row map is unchanged (plan identical on
# every rank, rank R gets item t x 64 + R); only rank -> host differs (ranks 0-7 on slinky-1 instead of slinky-10, etc.).
# Step 0 (before training): every node verifies the 8 staged stores (/scratch/amartyavarma/aloe-pose-ft/stores/pK, node digest == STORES.json);
# a node missing one copies it from the Weka build (stores/pK, = stage.sbatch's copy + per-file size check), then re-verifies.
# usage: QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 sbatch -w <8 nodes> --cpus-per-task=5 --mem=540G aloe_nopose.sbatch
# Stop = save-and-exit: touch aloe-nopose-ft/control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>).
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-nopose-ft; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
: "${QQA_INIT:?}"
set -a; source ${QQA_ENV:-$Q/aloe_nopose.env}; set +a
export QQA_INIT MASTER_PORT=29843
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$QQA_POSE_B" = 0 ] || { echo "ABORT: this is the no-pose ablation (QQA_POSE_B must be 0)"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe_pose.py $C/train_aloe_pose.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-nopose-ft job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * SLURM_NNODES))) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|PPM_CD|PPM_DEC_ATTN|S3_MIN)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
# ---- stores: verify on every node, stage from Weka where missing / different
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=5 --cpu-bind=none bash -c '
S=/data/home/amartyavarma/ppm/aloe-pose-ft/stores; D=/scratch/amartyavarma/aloe-pose-ft/stores; H=$(hostname -s); PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
chk() { S3_MIN_FRAMES=125 PPM_CD=2 $PY -I -c "
import sys, json; sys.path.insert(0, \"/data/home/amartyavarma/ppm/aloe-nopose-ft/code/lib\"); import s3rows as F
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
echo "$H all 8 stores verified (digests == STORES.json + every file name / size == the Weka build); $(df -h /scratch | tail -1)"' || { echo "ABORT: store staging failed"; exit 1; }
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_pose.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
