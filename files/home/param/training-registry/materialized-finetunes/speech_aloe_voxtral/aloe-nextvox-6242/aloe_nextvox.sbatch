#!/bin/bash
#SBATCH -J aloe-nextvox -p all -N8 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=7 --mem=540G -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-nextvox-6242/logs/aloe-nextvox-%j.log
#SBATCH --signal=B:USR1@300
# aloe-nextvox-6242 (10-18) = aloe-v2vox-raw/aloe_v2vox_raw.sbatch (job 23081, the PPM Voxtral-on-A Aloe FT) on the 14B next-arch w-6242 with:
#   * 8 nodes = 22420's 64 ranks one-to-one (ALOE_LNODES 0;1;...;7, one micro slot per GPU per step);
#   * init = the 14B export w-6242 (QQA_INIT default below), trainer code/train_aloe_nextvox.py (+ model_ilp / model_ilp_vox / rope_frame compile
#     checks), env aloe_nextvox.env, port 29873;
#   * the same pre-flight per node: the 4 stores node-local (aloe-v2vox-raw/tools/stage_raw.py --stores-only, digests), ALOE_FLAC_ROOT (22420's
#     path) -> symlink to the shared /data copy of all 8 logical nodes' A tracks (ledgers checked); local export root node-local if /scratch
#     has >= 60 GB free (14B: 6 exports x 4.2 GB per node; the 39B reference used 200 GB), else a symlink to /data/.../node-exports/<host>;
#     /data must keep >= 300 GB + the final export + possible node-export symlinks; GPU clocks printed (= aloe-next-6242's check).
# usage: sbatch -w <8 hosts> aloe_nextvox.sbatch      Stop = save-and-exit: touch control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>).
set -uo pipefail
[ "${SLURM_CPUS_PER_TASK:-0}" -ge 4 ] || { echo "ABORT: --cpus-per-task=${SLURM_CPUS_PER_TASK:-unset} < 4"; exit 1; }
[ "${SLURM_CPUS_PER_TASK}" -ge 8 ] || echo "NOTE: --cpus-per-task=${SLURM_CPUS_PER_TASK} (22420 had 8; 23081 had 5; engineering only)"
Q=/data/home/amartyavarma/ppm/aloe-nextvox-6242; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
export QQA_INIT=${QQA_INIT:-/data/home/amartyavarma/ppm/next-arch/ckpt-weka/ppm-next/w-6242}
set -a; source ${QQA_ENV:-$Q/aloe_nextvox.env}; set +a
export QQA_INIT MASTER_PORT=29873
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$VOX_A_GAIN" = raw ] || { echo "ABORT: the reference run is VOX_A_GAIN=raw"; exit 1; }
[ -f $QQA_INIT/COMPLETE ] || { echo "ABORT: init not COMPLETE"; exit 1; }
[ -f $VOX_ADAPTER/COMPLETE ] || { echo "ABORT: adapter export not COMPLETE"; exit 1; }
for f in $C/model_ppm.py $C/model_ilp.py $C/model_ilp_vox.py $C/rope_frame.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe.py $C/data_aloe_v2.py \
         $C/data_aloe_v2vox.py $C/train_aloe_nextvox.py $C/rowcache_vox.py $C/lib/audio_s4q.py $C/lib/voxtral_front.py $C/lib/vadapter_s4q.py; do
  $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
NN=$SLURM_NNODES
[ "$NN" -eq 8 ] || { echo "ABORT: $NN nodes (8 = 22420's 64 ranks one-to-one)"; exit 1; }
export ALOE_LNODES="0;1;2;3;4;5;6;7"
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-nextvox job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * NN)), ALOE_LNODES $ALOE_LNODES, cpus/task $SLURM_CPUS_PER_TASK) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|VOX_|S4_VOXTRAL|PPM_|ILP_|S3_MIN)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ "$FD" -ge $((300 + 40 + 30 * NN)) ] || { echo "ABORT: /data free ${FD} GB < 300 floor + 40 final export + 30 GB per node of possible node-export symlinks"; exit 1; }
srun -N$NN -n$NN --ntasks-per-node=1 -c 1 --mem=1G --gres=gpu:8 bash -c 'nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,memory.used,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown --format=csv,noheader | sed "s/^/$(hostname -s) gpu /"' | sort -V
# pre-flight on every node (one task per node) = 23081's
srun -N$NN -n$NN --ntasks-per-node=1 -c 4 --mem=32G --gres=none --kill-on-bad-exit=1 bash -c '
H=$(hostname -s); I=$(echo "$PPM_HOSTS" | tr , "\n" | grep -nx "$H" | cut -d: -f1); P=$((I - 1)); LN=$(echo "$ALOE_LNODES" | cut -d";" -f$I)
cd /tmp
OMP_NUM_THREADS=4 '"$PY"' -I /data/home/amartyavarma/ppm/aloe-v2vox-raw/tools/stage_raw.py --lnodes "" --root "$ALOE_SHARED_FLAC" --stores-only > '"$Q"'/logs/preflight-stores-$SLURM_JOB_ID-$H.log 2>&1 \
  || { echo "ABORT $H: store staging failed"; tail -3 '"$Q"'/logs/preflight-stores-$SLURM_JOB_ID-$H.log; exit 1; }
FR=$ALOE_FLAC_ROOT
if [ -L "$FR" ]; then [ "$(readlink "$FR")" = "$ALOE_SHARED_FLAC" ] || { echo "ABORT $H: $FR -> $(readlink "$FR") != $ALOE_SHARED_FLAC"; exit 1; }
elif [ -e "$FR" ]; then echo "ABORT $H: $FR exists as a real path (expected the symlink to the shared copy)"; exit 1
else mkdir -p "$(dirname "$FR")" && ln -s "$ALOE_SHARED_FLAC" "$FR" || { echo "ABORT $H: cannot link $FR"; exit 1; }; fi
for l in $(echo "$LN" | tr , " "); do
  [ -f "$FR/_ledger/DONE_v2vox_node$l.json" ] || { echo "ABORT $H: logical node $l A tracks not staged / verified"; exit 1; }; done
ER=$ALOE_LOCAL_EXPORT_ROOT; F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
if [ -L "$ER" ] || [ -d "$ER" ]; then :
elif [ "$F" -ge 60 ]; then mkdir -p "$ER"
else T=/data/home/amartyavarma/ppm/aloe-nextvox-6242/node-exports/$H; mkdir -p "$T" "$(dirname "$ER")" && ln -s "$T" "$ER"; fi
find /dev/shm -maxdepth 1 -user $USER \( -name "nccl-*" -o -name "torch_*" \) -mmin +5 -delete 2>/dev/null
echo "$H physical node $P: logical nodes $LN; stores ok; A tracks $(readlink $FR) ledgers ok; exports $( [ -L $ER ] && echo "-> $(readlink $ER)" || echo "node-local $ER"); /scratch free ${F} GB; /dev/shm $(df -h --output=used /dev/shm | tail -1)"' || exit 1
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_nextvox.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
