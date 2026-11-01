#!/bin/bash
#SBATCH -J aloev2vraw -p all --ntasks-per-node=8 --gres=gpu:8 -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2vox-raw/logs/aloev2vraw-%j.log
#SBATCH --signal=B:USR1@300
# aloe-v2vox-raw trainer (code/train_aloe_v2vox_raw.py + data_aloe_v2vox VOX_A_GAIN=raw) = aloe-v2vox-ft/aloe_v2vox.sbatch (job 22420) with:
#   * the raw-A env (aloe_v2vox_raw.env), 22420's 64 ranks as LOGICAL ranks on N physical nodes (gradient accumulation, same rows per step):
#     ALOE_LNODES from N (3 -> 0,1,2;3,4,5;6,7, 2 -> 0,1,2,3;4,5,6,7, 4 -> 0,1;2,3;4,5;6,7, 8 -> one each), physical node P = P-th host sorted -V;
#   * pre-flight per node: the 4 stores node-local (stage_raw.py --stores-only, digests), ALOE_FLAC_ROOT (22420's path) -> symlink to the shared
#     /data copy of all 8 logical nodes' A tracks (ledgers checked), local export root node-local if /scratch has >= 200 GB free, else a symlink
#     to /data/.../node-exports/<host> (the spare nodes' /scratch is nearly full); /data must keep >= 300 GB + the exports.
# usage: QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 sbatch -N<n> -w <hosts> --cpus-per-task=<c> --mem=540G aloe_v2vox_raw.sbatch
# Stop = save-and-exit: touch aloe-v2vox-raw/control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>): weights-only export at the next step boundary, then exit.
set -uo pipefail
[ "${SLURM_CPUS_PER_TASK:-0}" -ge 4 ] || { echo "ABORT: --cpus-per-task=${SLURM_CPUS_PER_TASK:-unset} < 4"; exit 1; }
[ "${SLURM_CPUS_PER_TASK}" -ge 8 ] || echo "NOTE: --cpus-per-task=${SLURM_CPUS_PER_TASK} (22420 had 8; the spare nodes' page / service jobs hold CPUs; engineering only)"
Q=/data/home/amartyavarma/ppm/aloe-v2vox-raw; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
: "${QQA_INIT:?}"
set -a; source ${QQA_ENV:-$Q/aloe_v2vox_raw.env}; set +a
export QQA_INIT MASTER_PORT=29871
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$VOX_A_GAIN" = raw ] || { echo "ABORT: this run is VOX_A_GAIN=raw"; exit 1; }
[ -f $QQA_INIT/COMPLETE ] || { echo "ABORT: init not COMPLETE"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe.py $C/data_aloe_v2.py $C/data_aloe_v2vox.py $C/train_aloe_v2vox_raw.py $C/rowcache_vox.py $C/lib/audio_s4q.py $C/lib/voxtral_front.py $C/lib/vadapter_s4q.py; do
  $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
NN=$SLURM_NNODES
if [ -z "${ALOE_LNODES:-}" ]; then
  case $NN in 2) ALOE_LNODES="0,1,2,3;4,5,6,7";; 3) ALOE_LNODES="0,1,2;3,4,5;6,7";; 4) ALOE_LNODES="0,1;2,3;4,5;6,7";; 8) ALOE_LNODES="0;1;2;3;4;5;6;7";;
    *) echo "ABORT: $NN nodes (2, 3, 4 or 8 only: 1 node does not fit the 39B FSDP state)"; exit 1;; esac
fi
export ALOE_LNODES
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-v2vox-raw job $SLURM_JOB_ID on $PPM_HOSTS (physical world $((8 * NN)), logical world 64, ALOE_LNODES $ALOE_LNODES, cpus/task $SLURM_CPUS_PER_TASK) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|VOX_|S4_VOXTRAL|PPM_CD|PPM_DEC_ATTN|S3_MIN)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ "$FD" -ge $((300 + 80 + 160 * NN)) ] || { echo "ABORT: /data free ${FD} GB < 300 floor + 80 final export + 160 GB per node of possible node-export symlinks"; exit 1; }
# pre-flight on every node (one task per node)
srun -N$NN -n$NN --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
H=$(hostname -s); I=$(echo "$PPM_HOSTS" | tr , "\n" | grep -nx "$H" | cut -d: -f1); P=$((I - 1)); LN=$(echo "$ALOE_LNODES" | cut -d";" -f$I)
cd /tmp
OMP_NUM_THREADS=4 '"$PY"' -I '"$Q"'/tools/stage_raw.py --lnodes "" --root "$ALOE_SHARED_FLAC" --stores-only > '"$Q"'/logs/preflight-stores-$SLURM_JOB_ID-$H.log 2>&1 \
  || { echo "ABORT $H: store staging failed"; tail -3 '"$Q"'/logs/preflight-stores-$SLURM_JOB_ID-$H.log; exit 1; }
FR=$ALOE_FLAC_ROOT
if [ -L "$FR" ]; then [ "$(readlink "$FR")" = "$ALOE_SHARED_FLAC" ] || { echo "ABORT $H: $FR -> $(readlink "$FR") != $ALOE_SHARED_FLAC"; exit 1; }
elif [ -e "$FR" ]; then echo "ABORT $H: $FR exists as a real path (expected the symlink to the shared copy)"; exit 1
else mkdir -p "$(dirname "$FR")" && ln -s "$ALOE_SHARED_FLAC" "$FR" || { echo "ABORT $H: cannot link $FR"; exit 1; }; fi
for l in $(echo "$LN" | tr , " "); do
  [ -f "$FR/_ledger/DONE_v2vox_node$l.json" ] || { echo "ABORT $H: logical node $l A tracks not staged / verified"; exit 1; }; done
ER=$ALOE_LOCAL_EXPORT_ROOT; F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
if [ -L "$ER" ] || [ -d "$ER" ]; then :
elif [ "$F" -ge 200 ]; then mkdir -p "$ER"
else T=/data/home/amartyavarma/ppm/aloe-v2vox-raw/node-exports/$H; mkdir -p "$T" "$(dirname "$ER")" && ln -s "$T" "$ER"; fi
find /dev/shm -maxdepth 1 -user $USER \( -name "nccl-*" -o -name "torch_*" \) -mmin +5 -delete 2>/dev/null
echo "$H physical node $P: logical nodes $LN; stores ok; A tracks $(readlink $FR) ledgers ok; exports $( [ -L $ER ] && echo "-> $(readlink $ER)" || echo "node-local $ER"); /scratch free ${F} GB; /dev/shm $(df -h --output=used /dev/shm | tail -1)"' || exit 1
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_v2vox_raw.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
