#!/bin/bash
#SBATCH -J aloevoxft -p all -N8 --ntasks-per-node=8 --gres=gpu:8 -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-vox-ft/logs/aloevoxft-%j.log
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:USR1@300
# aloe-vox-ft trainer (code/train_aloe_vox.py + data_aloe_vox), world 64 on the 8 prepared PPM hosts (node i = store pi + its A FLACs):
#   slinky-0,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-12 (sorted -V = rank order; asserted below).
# usage (launch.sh does this): QQA_INIT=<ppm-local-bf16-shards dir | consolidated bf16 DCP> sbatch -w <the 8 hosts> --cpus-per-task=8 --mem=540G aloe_vox.sbatch
# Stop = save-and-exit: touch aloe-vox-ft/control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>): weights-only export at the next step boundary, then exit.
set -uo pipefail
[ "${SLURM_CPUS_PER_TASK:-0}" -ge 8 ] || { echo "ABORT: --cpus-per-task=${SLURM_CPUS_PER_TASK:-unset} < 8 (SPEED_FIX.md: at 5 logical CPUs = 2.5 cores per rank the data workers + ffmpeg starve the FSDP forward)"; exit 1; }
Q=/data/home/amartyavarma/ppm/aloe-vox-ft; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
: "${QQA_INIT:?}"
set -a; source ${QQA_ENV:-$Q/aloe_vox.env}; set +a
export QQA_INIT MASTER_PORT=29851
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe_pose.py $C/data_aloe_vox.py $C/train_aloe_vox.py $C/lib/audio_s4q.py $C/lib/voxtral_front.py $C/lib/vadapter_s4q.py; do
  $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
WANT=${ALOE_VOX_HOSTS:-slinky-0,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-12}
[ "$PPM_HOSTS" = "$WANT" ] || { echo "ABORT: hosts $PPM_HOSTS != the prepared $WANT (node i must hold store p<i> + its FLACs)"; exit 1; }
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-vox-ft job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * SLURM_NNODES))) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|VOX_|S4_VOXTRAL|PPM_CD|PPM_DEC_ATTN|S3_MIN)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
# pre-flight on every node: its store + VOX index + verified FLAC ledger present, /dev/shm free of stale segments, scratch headroom for the exports
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
H=$(hostname -s); I=$(echo "$PPM_HOSTS" | tr , "\n" | grep -nx "$H" | cut -d: -f1); K=$((I - 1))
[ -f $ALOE_STORE_ROOT/p$K/VOX_DONE ] && [ -f $ALOE_FLAC_ROOT/_ledger/DONE_store$K.json ] || { echo "ABORT $H: store p$K / FLAC ledger missing"; exit 1; }
find /dev/shm -maxdepth 1 -user $USER \( -name "nccl-*" -o -name "torch_*" \) -mmin +5 -delete 2>/dev/null
F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); echo "$H node $K: store p$K + FLACs ok; rowcache $(grep -o coverage.:.[0-9.]* $ALOE_ROWCACHE/LEDGER_node$K.json 2>/dev/null || echo none); /scratch free ${F} GB; /dev/shm $(df -h --output=used /dev/shm | tail -1)"
[ "$F" -ge 70 ] || { echo "ABORT $H: /scratch free ${F} GB < 70 (6 local exports x ~10.3 GB)"; exit 1; }' || exit 1
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_vox.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
