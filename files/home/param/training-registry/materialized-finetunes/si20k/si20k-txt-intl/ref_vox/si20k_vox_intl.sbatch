#!/bin/bash
#SBATCH -J si20kvox-intl -p all -N8 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=7 --mem=540G -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/si20k-vox-intl/logs/si20kvox-intl-%j.log
#SBATCH --signal=B:USR1@300
# si20k-vox-intl (10-19) = aloe-nextvox-6242/aloe_nextvox.sbatch (job 24193, 14B next-arch w-6242 + 14B ASR adapter w-480) with:
#   * the si20k data (env si20k_vox_intl.env + plan/PLAN.env from finalize_plan.py), trainer code/train_si20k_vox.py, port 29883;
#   * 8 nodes = 64 ranks one-to-one (ALOE_LNODES 0;1;...;7), = 24193's layout;
#   * pre-flight per node (si20k-vox-ppm/tools/stage_si20k.py with SI20K_CODE = this run's code): the 4 si20k stores node-local (digests),
#     the plan digest, every logical node's A FLACs verified; node-local exports REQUIRED (/scratch >= 40 GB free after the stores: 6 x ~3.6 GB
#     per node + margin); /data >= 300 GB floor + 40 GB final export; GPU clocks printed (= 24193's check).
# usage: sbatch -w <8 hosts> si20k_vox_intl.sbatch     Stop = save-and-exit: touch control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>).
set -uo pipefail
[ "${SLURM_CPUS_PER_TASK:-0}" -ge 3 ] || { echo "ABORT: --cpus-per-task=${SLURM_CPUS_PER_TASK:-unset} < 3"; exit 1; }
Q=/data/home/amartyavarma/ppm/si20k-vox-intl; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
export QQA_INIT=${QQA_INIT:-/data/home/amartyavarma/ppm/next-arch/ckpt-weka/ppm-next/w-6242}
[ -f $Q/plan/PLAN.env ] || { echo "ABORT: plan/PLAN.env missing (run finalize_plan.py after the store build)"; exit 1; }
set -a; source ${QQA_ENV:-$Q/si20k_vox_intl.env}; source $Q/plan/PLAN.env; set +a
export QQA_INIT MASTER_PORT=29883 SI20K_CODE=$C
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$VOX_A_GAIN" = raw ] || { echo "ABORT: the reference run is VOX_A_GAIN=raw"; exit 1; }
[ "$(basename $VOX_ADAPTER)" = adapter-asr14b-w480 ] || echo "NOTE: VOX_ADAPTER $VOX_ADAPTER (24193 used aloe-nextvox-6242/adapter-asr14b-w480; sha asserted at init)"
[ -f $QQA_INIT/COMPLETE ] || { echo "ABORT: init not COMPLETE"; exit 1; }
[ -f $VOX_ADAPTER/COMPLETE ] || { echo "ABORT: adapter export not COMPLETE"; exit 1; }
[ -f $ALOE_SI20K_STORES_JSON ] || { echo "ABORT: $ALOE_SI20K_STORES_JSON missing"; exit 1; }
$PY -I -c "import json,sys; j=json.load(open(sys.argv[1])); assert j['plan']['digest']==sys.argv[2] and j['plan']['steps']==int(sys.argv[3]), (j['plan']['digest'], j['plan']['steps']); assert j.get('keep_spans') or j.get('keep_list') or sys.argv[4]=='1', 'stores built without the filter keep list / spans'" \
  $ALOE_SI20K_STORES_JSON $QQA_REF_PLAN_DIGEST $QQA_STEPS ${SI20K_ALLOW_NO_FILTER:-0} || { echo "ABORT: STORES.json != plan/PLAN.env (or no filter)"; exit 1; }
for f in $C/model_ppm.py $C/model_ilp.py $C/model_ilp_vox.py $C/rope_frame.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe_v2vox.py $C/data_si20k_vox.py $C/train_si20k_vox.py $C/lib/audio_s4q.py $C/lib/voxtral_front.py $C/lib/vadapter_s4q.py; do
  $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
NN=$SLURM_NNODES
[ "$NN" -eq 8 ] || { echo "ABORT: $NN nodes (8 = the 64 logical ranks one-to-one)"; exit 1; }
case ",$PPM_HOSTS," in *,slinky-4,*|*,slinky-18,*) echo "ABORT: slinky-4 / slinky-18 GPUs are reserved (pose investigation)"; exit 1;; esac
export ALOE_LNODES="0;1;2;3;4;5;6;7"
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== si20k-vox-intl job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * NN)), ALOE_LNODES $ALOE_LNODES, cpus/task $SLURM_CPUS_PER_TASK) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN steps $QQA_STEPS plan $QQA_REF_PLAN_DIGEST"
env | grep -E '^(QQA_|ALOE_|VOX_|SI20K|S4_VOXTRAL|PPM_|ILP_|S3_MIN)' | sort
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ "$FD" -ge $((300 + 10)) ] || { echo "ABORT: /data free ${FD} GB < 300 floor + 10"; exit 1; }
[ "$FD" -ge $((300 + 40)) ] || echo "NOTE: /data free ${FD} GB < 300 + 40 GB final export: the watchers free local-12829 and the /data si20k stores after step 1; the trainer waits up to QQA_EXPORT_WAIT_MIN for space"
srun -N$NN -n$NN --ntasks-per-node=1 -c 1 --mem=1G --gres=gpu:8 bash -c 'nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,memory.used,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown --format=csv,noheader | sed "s/^/$(hostname -s) gpu /"' | sort -V
# pre-flight on every node (one task per node)
srun -N$NN -n$NN --ntasks-per-node=1 -c 4 --mem=32G --gres=none --kill-on-bad-exit=1 bash -c '
H=$(hostname -s); I=$(echo "$PPM_HOSTS" | tr , "\n" | grep -nx "$H" | cut -d: -f1); P=$((I - 1)); LN=$(echo "$ALOE_LNODES" | cut -d";" -f$I)
cd /tmp
OMP_NUM_THREADS=4 '"$PY"' -I /data/home/amartyavarma/ppm/si20k-vox-ppm/tools/stage_si20k.py --lnodes "$LN" --min-scratch-gb 24 > '"$Q"'/logs/preflight-$SLURM_JOB_ID-$H.log 2>&1 \
  || { echo "ABORT $H: si20k staging / A ledger check failed"; tail -3 '"$Q"'/logs/preflight-$SLURM_JOB_ID-$H.log; exit 1; }
ER=$ALOE_LOCAL_EXPORT_ROOT; F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
[ -L "$ER" ] && { echo "ABORT $H: $ER is a symlink (node-local exports required)"; exit 1; }
[ "$F" -ge 24 ] || { echo "ABORT $H: /scratch free ${F} GB < 24 (node-local exports)"; exit 1; }
mkdir -p "$ER"
find /dev/shm -maxdepth 1 -user $USER \( -name "nccl-*" -o -name "torch_*" \) -mmin +5 -delete 2>/dev/null
echo "$H physical node $P: logical nodes $LN; si20k stores + A ledgers ok ($(tail -1 '"$Q"'/logs/preflight-$SLURM_JOB_ID-$H.log | cut -c1-120)); exports node-local $ER; /scratch free ${F} GB; /dev/shm $(df -h --output=used /dev/shm | tail -1)"' || exit 1
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_si20k_vox.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
