#!/bin/bash
#SBATCH -J aloepersona -p all --ntasks-per-node=8 --gres=gpu:8 -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/logs/aloepersona-%j.log
#SBATCH --signal=B:USR1@300
# [persona] (10-24) = vox/run_vox.sbatch with ONLY: job name aloepersona, log persona/logs, code dir persona/code (snapshot of job 29345's code +
#   data_aloe_persona.py + train_samsep_persona.py), QQA_ENV default persona/run_persona.env (exported for hf_watch.sh). usage: bash persona/launch_persona.sh
# w64 (10-24, user via main: 8 hosts x 8 GPUs = world 64, the reference layout, 1 slot / GPU, no host offload) = run.sbatch with: N-host
#   preflight / watcher steps (-N$SLURM_NNODES), and stores read straight from the /data sam-sep source when ALOE_STORES_FROM_DATA=1 (same
#   digests, asserted on load; no node-local copy: the MoT hosts' /scratch has no room for it next to the init + exports + floor).
# aloe-ft-32-samsep (10-24): Aloe FT, NO cotrain, sam-sep Mimi codes (README in run.env; CFG_DIFF.txt) = aloe-zoom-cotrain/aloe_zoom_cotrain.sbatch
#   (job 26467) with: the aloe-only trainer mode (code/train_samsep.py), world 14 on 2 hosts (slinky-18 x 8 + slinky-4 GPUs 0-5 x 6: two
#   per-host srun steps of code/task_samsep.sh, rank = host offset + local id), per-host pre-flight for the sam-sep stores / A FLAC root /
#   init copy, and the HF watcher (hf_watch.sh) as a nice'd --cpu-bind=none overlap step on both hosts (each node-local export -> private HF
#   as soon as it is written; intermediates deleted after VERIFY PASS when HFLX_DELETE_INTER=1). Never slinky-10 (IB rail 7) in this job.
# usage: bash launch.sh   (= sbatch -N2 -w slinky-4,slinky-18 --ntasks-per-node=8 --cpus-per-task=5 --mem=540G run.sbatch)
# Stop = save-and-exit: touch control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>): node-local full state (if /scratch has room) +
#   weights-only export-stop<N>, then exit 0. Resume: COTRAIN_RESUME=<saves>/<run>/dcp-<N> bash launch.sh (same 2 hosts, same layout).
set -uo pipefail
[ "${SLURM_CPUS_PER_TASK:-0}" -ge 4 ] || { echo "ABORT: --cpus-per-task=${SLURM_CPUS_PER_TASK:-unset} < 4"; exit 1; }
Q=/data/home/amartyavarma/ppm/aloe-ft-32-samsep; C=$Q/persona/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
export QQA_ENV=${QQA_ENV:-$Q/persona/run_persona.env}   # [persona] exported: hf_watch.sh sources it too
set -a; source $QQA_ENV; set +a
export QQA_INIT MASTER_PORT=29877 COTRAIN_RESUME=${COTRAIN_RESUME:-}
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$VOX_A_GAIN" = raw ] || { echo "ABORT: this run is VOX_A_GAIN=raw"; exit 1; }
[ "$COTRAIN_GROUP_MODE" = aloe_only ] || { echo "ABORT: no cotrain (COTRAIN_GROUP_MODE=aloe_only)"; exit 1; }
[ -f $VOX_ADAPTER/COMPLETE ] || { echo "ABORT: ASR adapter export not COMPLETE"; exit 1; }
for f in $COTRAIN_REF_CFG $ALOE_REF_CFG_26467 $ALOE_M1VOX_INDEX $ALOE_RAWGAIN $ALOE_SAMSEP_ROW_MAP; do [ -f $f ] || { echo "ABORT: missing $f"; exit 1; }; done
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe.py $C/data_aloe_v2.py $C/data_aloe_v2vox.py $C/safety_ppm.py $C/optoff.py \
         $C/train_samsep_persona.py $C/data_aloe_persona.py $C/rowcache_vox.py $C/lib/audio_s4q.py $C/lib/voxtral_front.py $C/lib/vadapter_s4q.py; do
  $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
ALLOC=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort | paste -sd,)
[ "$ALLOC" = "$(echo $PPM_HOSTS | tr , '\n' | sort | paste -sd,)" ] || { echo "ABORT: allocation $ALLOC != PPM_HOSTS $PPM_HOSTS"; exit 1; }
echo ",$ALLOC," | grep -q ",slinky-10," && { echo "ABORT: slinky-10 never in a multi-node job (IB rail 7)"; exit 1; }
export PPM_HOSTS PPM_HOST_NRANKS PPM_HOST_GPUS
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR $Q/logs; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-ft-32-samsep job $SLURM_JOB_ID on $PPM_HOSTS (ranks $PPM_HOST_NRANKS, GPUs '$PPM_HOST_GPUS'; logical world $ALOE_LOGICAL_WORLD; cpus/task $SLURM_CPUS_PER_TASK) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN resume '${COTRAIN_RESUME}'"
env | grep -E '^(QQA_|ALOE_|VOX_|COTRAIN_|PPM_HOST|S4_VOXTRAL|PPM_CD|PPM_DEC_ATTN|S3_MIN|HF_REPO|HFLX_)' | sort
FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ "$FD" -ge $((300 + 80)) ] && echo "/data free ${FD} GB: final consolidated /data export will be attempted (floor 300)" \
  || echo "NOTE: /data free ${FD} GB < 300 floor + 80: the final consolidated /data export will be REFUSED (node-local export-<last> + HF backup = final copy)"
# pre-flight on every host (one task per host): sam-sep stores node-local (DATA's stage script: copy or cmp-verify), the A FLAC root symlink +
# all 8 ledgers of the new plan, the init copy, /scratch room, stale /dev/shm NCCL files, the GPUs this host will use
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
H=$(hostname -s); cd /tmp
if [ "${ALOE_STORES_FROM_DATA:-0}" = 1 ]; then [ "$ALOE_SAMSEP_STORES" = "$ALOE_SAMSEP_SRC" ] || { echo "ABORT $H: ALOE_STORES_FROM_DATA=1 but stores != src"; exit 1; }
  echo "stores from /data (no staging) verified" > '"$Q"'/logs/preflight-stores-$SLURM_JOB_ID-$H.log
else bash '"$Q"'/data/tools/stage_samsep_stores.sh > '"$Q"'/logs/preflight-stores-$SLURM_JOB_ID-$H.log 2>&1 \
  || { echo "ABORT $H: store staging / verify failed"; tail -3 '"$Q"'/logs/preflight-stores-$SLURM_JOB_ID-$H.log; exit 1; }; fi
FR=$ALOE_FLAC_ROOT
if [ -L "$FR" ]; then [ "$(readlink "$FR")" = "$ALOE_SHARED_FLAC" ] || { echo "ABORT $H: $FR -> $(readlink "$FR") != $ALOE_SHARED_FLAC"; exit 1; }
elif [ -e "$FR" ]; then echo "ABORT $H: $FR exists as a real path (expected the symlink to the shared copy)"; exit 1
else mkdir -p "$(dirname "$FR")" && ln -s "$ALOE_SHARED_FLAC" "$FR" || { echo "ABORT $H: cannot link $FR"; exit 1; }; fi
for l in 0 1 2 3 4 5 6 7; do grep -q "\"plan_digest\": \"$QQA_REF_PLAN_DIGEST\"" "$FR/_ledger/DONE_v2vox_node$l.json" 2>/dev/null \
  || { echo "ABORT $H: logical node $l A tracks not verified for plan $QQA_REF_PLAN_DIGEST"; exit 1; }; done
[ -f "$QQA_INIT/COMPLETE" ] && { [ -f "$QQA_INIT/.metadata" ] || [ -f "$QQA_INIT/meta.json" ]; } || { echo "ABORT $H: init $QQA_INIT not COMPLETE on this host"; exit 1; }
F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
[ "$F" -ge $((ALOE_EXPORT_SCRATCH_FLOOR_GB + 10)) ] || { echo "ABORT $H: /scratch free ${F} GB < floor $ALOE_EXPORT_SCRATCH_FLOOR_GB + 10"; exit 1; }
mkdir -p "$ALOE_LOCAL_EXPORT_ROOT" "$COTRAIN_RESUME_ROOT" /scratch/amartyavarma/aloe-ft-32-samsep
if [ -n "$COTRAIN_RESUME" ]; then [ -f "$COTRAIN_RESUME/COMPLETE" ] || { echo "ABORT $H: resume state $COTRAIN_RESUME not COMPLETE on this host"; exit 1; }; fi
find /dev/shm -maxdepth 1 -user $USER \( -name "nccl-*" -o -name "torch_*" \) -mmin +5 -delete 2>/dev/null
I=$(echo "$PPM_HOSTS" | tr , "\n" | grep -nx "$H" | cut -d: -f1); G=$(echo "$PPM_HOST_GPUS" | cut -d";" -f$I)
echo "$H: stores ok ($(grep -c "verified\|staged + verified" '"$Q"'/logs/preflight-stores-$SLURM_JOB_ID-$H.log) lines), A tracks $(readlink $FR) ledgers ok, init ok; /scratch free ${F} GB; GPUs $G: $(nvidia-smi -i $G --query-gpu=index,temperature.gpu,clocks.sm,clocks.max.sm --format=csv,noheader | tr "\n" " ")"' || exit 1
# training: one srun step per host (uneven rank counts), rank = host offset + local id (code/task_samsep.sh)
PIDS=()
IFS=, read -ra HS <<< "$PPM_HOSTS"; IFS=, read -ra NR <<< "$PPM_HOST_NRANKS"
for i in "${!HS[@]}"; do
  srun -N1 -n${NR[$i]} -w ${HS[$i]} --ntasks-per-node=${NR[$i]} --kill-on-bad-exit=1 bash $C/task_samsep.sh $C/train_samsep_persona.py &
  PIDS+=($!)
done
# HF watcher (after the training steps hold their resources): one nice'd task per host, overlap step, no CPU binding, no GPU (never GCS)
sleep 20
srun --overlap -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c 2 --mem=16G --gres=none --cpu-bind=none --kill-on-bad-exit=0 bash $Q/hf_watch.sh > $Q/logs/hf_watch-$SLURM_JOB_ID.log 2>&1 &
HFW=$!
alive() { local n=0 p; for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null && n=$((n + 1)); done; echo $n; }
while [ "$(alive)" = "${#PIDS[@]}" ]; do sleep 15; done
# one host's step ended: a normal finish ends the other within seconds; otherwise (crash on one host) stop the other after 120 s
for i in $(seq 8); do [ "$(alive)" = 0 ] && break; sleep 15; done
[ "$(alive)" = 0 ] || { echo "== one host's step ended, the other still runs after 120 s: stopping it $(TZ=America/Los_Angeles date '+%T PT')"; for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done; }
rc=0; for p in "${PIDS[@]}"; do wait $p; r=$?; [ $r -ne 0 ] && rc=$r; done
echo "== training exit $rc $(TZ=America/Los_Angeles date '+%T PT')"
echo "$rc" > $QQA_CONTROL_DIR/TRAIN_EXITED-$SLURM_JOB_ID
# let the HF watcher finish the remaining uploads (final export included), at most 3 h
for i in $(seq 720); do kill -0 $HFW 2>/dev/null || break; sleep 15; done
kill -0 $HFW 2>/dev/null && { echo "== HF watcher still running after 3 h: stopping it (re-run hf_watch.sh to finish)"; kill $HFW; }
wait $HFW; echo "== hf watcher exit $? $(TZ=America/Los_Angeles date '+%T PT')"
exit $rc
