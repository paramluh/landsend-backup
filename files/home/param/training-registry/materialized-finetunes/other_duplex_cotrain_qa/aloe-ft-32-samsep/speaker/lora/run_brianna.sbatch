#!/bin/bash
#SBATCH -J aloespklora -p all --ntasks-per-node=8 --gres=gpu:8 -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-%j.log
#SBATCH --signal=B:USR1@300
# aloe SPEAKER LoRA FT (10-24, PREP ONLY until the user's go) = speaker/run_speaker.sbatch with Q = speaker/lora, code = lora/code, trainer
#   train_spklora_vox.py, env run_brianna.env, ONE host (slinky-18). Below: run_speaker.sbatch's doc. = vox/run_vox.sbatch (job 29345) with: the speaker dirs (Q = speaker/, code =
#   speaker/code, trainer train_speaker_vox.py), env run_speaker.env, the per-plan A-FLAC ledgers (DONE_spk_<digest>_lw<LW>_node<l>, one per 8
#   logical ranks of THIS logical world), the init meta.json must carry vgain_theta_fp32, the speaker HF watcher copy. Same per-host training srun
#   steps (code/task_samsep.sh, rank = host offset + local id), same stop / save handling.
# usage: bash /data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/launch_brianna.sh
# Stop = save-and-exit: touch speaker/control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>).
set -uo pipefail
[ "${SLURM_CPUS_PER_TASK:-0}" -ge 4 ] || { echo "ABORT: --cpus-per-task=${SLURM_CPUS_PER_TASK:-unset} < 4"; exit 1; }
Q=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source ${QQA_ENV:-$Q/run_brianna.env}; set +a
export QQA_INIT MASTER_PORT=29893 COTRAIN_RESUME=${COTRAIN_RESUME:-}
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$VOX_A_GAIN" = raw ] || { echo "ABORT: this run is VOX_A_GAIN=raw"; exit 1; }
[ "$COTRAIN_GROUP_MODE" = aloe_only ] || { echo "ABORT: no cotrain (COTRAIN_GROUP_MODE=aloe_only)"; exit 1; }
[ -n "${ALOE_SPK_ACCOUNT:-}" ] || { echo "ABORT: ALOE_SPK_ACCOUNT unset"; exit 1; }
[ -f $VOX_ADAPTER/COMPLETE ] || { echo "ABORT: init / adapter export not COMPLETE"; exit 1; }
for f in $COTRAIN_REF_CFG $ALOE_REF_CFG_26467 $ALOE_REF_CFG_29345 $ALOE_M1VOX_INDEX $ALOE_RAWGAIN $ALOE_SAMSEP_ROW_MAP $ALOE_SPK_PLAN; do [ -f $f ] || { echo "ABORT: missing $f"; exit 1; }; done
for f in $C/model_ppm.py $C/model_lora.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe.py $C/data_aloe_v2.py $C/data_aloe_v2vox.py $C/data_aloe_spk.py $C/spk_plan.py \
         $C/safety_ppm.py $C/optoff.py $C/train_spklora_vox.py $C/rowcache_vox.py $C/lib/audio_s4q.py $C/lib/voxtral_front.py $C/lib/vadapter_s4q.py; do
  $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
ALLOC=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort | paste -sd,)
[ "$ALLOC" = "$(echo $PPM_HOSTS | tr , '\n' | sort | paste -sd,)" ] || { echo "ABORT: allocation $ALLOC != PPM_HOSTS $PPM_HOSTS"; exit 1; }
echo ",$ALLOC," | grep -q ",slinky-10," && { echo "ABORT: slinky-10 never in a multi-node job (IB rail 7)"; exit 1; }
export PPM_HOSTS PPM_HOST_NRANKS PPM_HOST_GPUS
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR $Q/logs; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe speaker FT job $SLURM_JOB_ID account $ALOE_SPK_ACCOUNT on $PPM_HOSTS (ranks $PPM_HOST_NRANKS; logical world $ALOE_LOGICAL_WORLD; cpus/task $SLURM_CPUS_PER_TASK) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN resume '${COTRAIN_RESUME}'"
env | grep -E '^(QQA_|ALOE_|VOX_|COTRAIN_|PPM_HOST|S4_VOXTRAL|PPM_CD|PPM_DEC_ATTN|S3_MIN|HF_REPO|HFLX_)' | sort
# pre-flight on every host: stores from /data, the speaker A-FLAC root symlink + this plan's ledgers, the init (COMPLETE + meta theta), /scratch room
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
H=$(hostname -s); cd /tmp
[ "${ALOE_STORES_FROM_DATA:-0}" = 1 ] && [ "$ALOE_SAMSEP_STORES" = "$ALOE_SAMSEP_SRC" ] || { echo "ABORT $H: the speaker FT reads the sam-sep stores from /data"; exit 1; }
FR=$ALOE_FLAC_ROOT
if [ -L "$FR" ]; then [ "$(readlink "$FR")" = "$ALOE_SHARED_FLAC" ] || { echo "ABORT $H: $FR -> $(readlink "$FR") != $ALOE_SHARED_FLAC"; exit 1; }
elif [ -e "$FR" ]; then echo "ABORT $H: $FR exists as a real path (expected the symlink to the shared copy)"; exit 1
else mkdir -p "$(dirname "$FR")" && ln -s "$ALOE_SHARED_FLAC" "$FR" || { echo "ABORT $H: cannot link $FR"; exit 1; }; fi
for l in $(seq 0 $(((ALOE_LOGICAL_WORLD + 7) / 8 - 1))); do grep -q "\"plan_digest\": \"$QQA_REF_PLAN_DIGEST\"" "$FR/_ledger/DONE_spk_${QQA_REF_PLAN_DIGEST}_lw${ALOE_LOGICAL_WORLD}_node$l.json" 2>/dev/null \
  || { echo "ABORT $H: logical node $l A tracks not verified for plan $QQA_REF_PLAN_DIGEST (lw $ALOE_LOGICAL_WORLD)"; exit 1; }; done
[ -f "$QQA_INIT/COMPLETE" ] && grep -q "\"vgain_theta_fp32\"" "$QQA_INIT/meta.json" || { echo "ABORT $H: init $QQA_INIT not COMPLETE or its meta.json has no vgain_theta_fp32"; exit 1; }
F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
[ "$F" -ge $((ALOE_EXPORT_SCRATCH_FLOOR_GB + 10)) ] || { echo "ABORT $H: /scratch free ${F} GB < floor $ALOE_EXPORT_SCRATCH_FLOOR_GB + 10"; exit 1; }
mkdir -p "$ALOE_LOCAL_EXPORT_ROOT" "$COTRAIN_RESUME_ROOT"
if [ -n "$COTRAIN_RESUME" ]; then [ -f "$COTRAIN_RESUME/COMPLETE" ] || { echo "ABORT $H: resume state $COTRAIN_RESUME not COMPLETE on this host"; exit 1; }; fi
find /dev/shm -maxdepth 1 -user $USER \( -name "nccl-*" -o -name "torch_*" \) -mmin +5 -delete 2>/dev/null
I=$(echo "$PPM_HOSTS" | tr , "\n" | grep -nx "$H" | cut -d: -f1); G=$(echo "$PPM_HOST_GPUS" | cut -d";" -f$I)
echo "$H: A tracks $(readlink $FR) ledgers ok, init ok ($(ls $QQA_INIT/rank*.pt | wc -l) rank files); /scratch free ${F} GB; GPUs $G: $(nvidia-smi -i $G --query-gpu=index,temperature.gpu,clocks.sm,clocks.max.sm --format=csv,noheader | tr "\n" " ")"' || exit 1
# training: one srun step per host, rank = host offset + local id (code/task_samsep.sh)
PIDS=()
IFS=, read -ra HS <<< "$PPM_HOSTS"; IFS=, read -ra NR <<< "$PPM_HOST_NRANKS"
for i in "${!HS[@]}"; do
  srun -N1 -n${NR[$i]} -w ${HS[$i]} --ntasks-per-node=${NR[$i]} --kill-on-bad-exit=1 bash $C/task_samsep.sh $C/train_spklora_vox.py &
  PIDS+=($!)
done
# HF watcher (speaker copy): one nice'd task per host, overlap step, no CPU binding, no GPU (private HF only, never GCS)
sleep 20
QQA_ENV=${QQA_ENV:-$Q/run_brianna.env} srun --overlap -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c 2 --mem=16G --gres=none --cpu-bind=none --kill-on-bad-exit=0 bash $Q/hf_watch.sh > $Q/logs/hf_watch-$SLURM_JOB_ID.log 2>&1 &
HFW=$!
alive() { local n=0 p; for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null && n=$((n + 1)); done; echo $n; }
while [ "$(alive)" = "${#PIDS[@]}" ]; do sleep 15; done
for i in $(seq 8); do [ "$(alive)" = 0 ] && break; sleep 15; done
[ "$(alive)" = 0 ] || { echo "== one host's step ended, the other still runs after 120 s: stopping it $(TZ=America/Los_Angeles date '+%T PT')"; for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done; }
rc=0; for p in "${PIDS[@]}"; do wait $p; r=$?; [ $r -ne 0 ] && rc=$r; done
echo "== training exit $rc $(TZ=America/Los_Angeles date '+%T PT')"
echo "$rc" > $QQA_CONTROL_DIR/TRAIN_EXITED-$SLURM_JOB_ID
for i in $(seq 720); do kill -0 $HFW 2>/dev/null || break; sleep 15; done
kill -0 $HFW 2>/dev/null && { echo "== HF watcher still running after 3 h: stopping it (re-run hf_watch.sh to finish)"; kill $HFW; }
wait $HFW; echo "== hf watcher exit $? $(TZ=America/Los_Angeles date '+%T PT')"
exit $rc
