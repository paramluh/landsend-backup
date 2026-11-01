#!/bin/bash
#SBATCH -J aloev2vox -p all -N8 --ntasks-per-node=8 --gres=gpu:8 -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2vox-ft/logs/aloev2vox-%j.log
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:USR1@300
# aloe-v2vox-ft trainer (code/train_aloe_v2vox.py + data_aloe_v2vox = v2's data + m1vox's channel A Voxtral) = aloe-m1vox-ft/aloe_m1vox.sbatch with the
# v2vox code / env, the v2-plan A-track ledger in the pre-flight. World 64 on slinky-0,2,3,5,6,8,9,12 (sorted -V = rank order = staging node order; asserted).
# usage: ALOE_VOX_HOSTS=<the 8 hosts sorted -V> QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 sbatch -w <hosts> --mem=540G aloe_v2vox.sbatch
# Stop = save-and-exit: touch aloe-v2vox-ft/control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>): weights-only export at the next step boundary, then exit.
set -uo pipefail
[ "${SLURM_CPUS_PER_TASK:-0}" -ge 8 ] || { echo "ABORT: --cpus-per-task=${SLURM_CPUS_PER_TASK:-unset} < 8 (m1vox SPEED_FIX.md)"; exit 1; }
Q=/data/home/amartyavarma/ppm/aloe-v2vox-ft; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
: "${QQA_INIT:?}"
set -a; source ${QQA_ENV:-$Q/aloe_v2vox.env}; set +a
export QQA_INIT MASTER_PORT=29861
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ -f $QQA_INIT/COMPLETE ] || { echo "ABORT: init not COMPLETE"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe.py $C/data_aloe_v2.py $C/data_aloe_v2vox.py $C/train_aloe_v2vox.py $C/rowcache_vox.py $C/lib/audio_s4q.py $C/lib/voxtral_front.py $C/lib/vadapter_s4q.py; do
  $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
WANT=${ALOE_VOX_HOSTS:?set ALOE_VOX_HOSTS to the 8 staged hosts (sorted -V)}
[ "$PPM_HOSTS" = "$WANT" ] || { echo "ABORT: hosts $PPM_HOSTS != the prepared $WANT (node i must hold the v2-plan A tracks of node i)"; exit 1; }
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-v2vox-ft job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * SLURM_NNODES))) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|VOX_|S4_VOXTRAL|PPM_CD|PPM_DEC_ATTN|S3_MIN)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
# pre-flight on every node: its 4 stores + the v2-plan A-track ledger present, /dev/shm free of stale segments, scratch headroom for the exports
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
H=$(hostname -s); I=$(echo "$PPM_HOSTS" | tr , "\n" | grep -nx "$H" | cut -d: -f1); K=$((I - 1))
for k in 0 1 2 3; do [ -f /scratch/amartyavarma/aloe-ft/stores/m1n$k/NODE.json ] || { echo "ABORT $H: store m1n$k not staged"; exit 1; }; done
[ -f $ALOE_FLAC_ROOT/_ledger/DONE_v2vox_node$K.json ] || { echo "ABORT $H: node $K v2-plan A tracks not staged / verified (tools/stage_v2vox.py)"; exit 1; }
find /dev/shm -maxdepth 1 -user $USER \( -name "nccl-*" -o -name "torch_*" \) -mmin +5 -delete 2>/dev/null
F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); echo "$H node $K: 4 stores + v2-plan A tracks ok ($(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[\"items\"], \"items\", d[\"gcs_files\"], \"FLACs\", d[\"mimidec\"], \"mimidec\")" $ALOE_FLAC_ROOT/_ledger/DONE_v2vox_node$K.json)); /scratch free ${F} GB; /dev/shm $(df -h --output=used /dev/shm | tail -1)"
[ "$F" -ge 100 ] || { echo "ABORT $H: /scratch free ${F} GB < 100 (keep >= 100 GB for the 6 local exports)"; exit 1; }' || exit 1
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_v2vox.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
