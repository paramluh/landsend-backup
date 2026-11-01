#!/bin/bash
#SBATCH -J pdxraw -p all --ntasks-per-node=8 --gres=gpu:8 -t 24:00:00
#SBATCH -o /data/home/amartyavarma/ppm/poseduplex-raw/logs/pdxraw-%j.log
#SBATCH --signal=B:USR1@300
# aloe-ft-32-samsep POSE-DUPLEX (10-25) = vox/run_vox.sbatch (job 29345) with: the pose-duplex trainer (code/train_samsep_posedup.py), its env
#   (run_poseduplex.env, hosts from launch_poseduplex.sh via POSEDUP_HOSTS), the extra pre-flight lines [pdx] (v5 init subset staged + its tag,
#   the pose index sha + a pose file readable, the plan digest set), the poseduplex HF watcher copy, and logs under poseduplex/logs.
# usage: bash launch_poseduplex.sh h0 h1 h2 h3 h4 h5 h6 h7   (8 x 8 GPUs, world 64; the init must be staged on these hosts first)
# Stop = save-and-exit: touch poseduplex/control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>): node-local full state (+ dec_p AdamW) if /scratch
#   has room + weights-only export-stop<N>, then exit 0. SAVE_NOW: full state, training continues. END: full save (ALOE_END_SAVE=1) + final export.
#   Resume: COTRAIN_RESUME=<saves>/<run>/dcp-<N> bash launch_poseduplex.sh <same 8 hosts, same order>.
set -uo pipefail
[ "${SLURM_CPUS_PER_TASK:-0}" -ge 4 ] || { echo "ABORT: --cpus-per-task=${SLURM_CPUS_PER_TASK:-unset} < 4"; exit 1; }
P=/data/home/amartyavarma/ppm/poseduplex-raw; C=$P/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source ${QQA_ENV:-$P/run_poseduplex.env}; set +a
export QQA_INIT MASTER_PORT=29881 COTRAIN_RESUME=${COTRAIN_RESUME:-}
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$VOX_A_GAIN" = raw ] || { echo "ABORT: this run is VOX_A_GAIN=raw"; exit 1; }
[ "$COTRAIN_GROUP_MODE" = hosts ] || { echo "ABORT: [raw] zoom co-train run (COTRAIN_GROUP_MODE=hosts)"; exit 1; }   # [raw] was aloe_only
[ "$POSEDUP_RAW_CODES" = 1 ] && [ "$POSE_BAND_MASK" = 1 ] && [ "$POSEDUP_ITEMS" = pose_ok ] && [ "${FRAMING_COND:-0}" = 0 ] || { echo "ABORT: [raw] needs POSEDUP_RAW_CODES=1 POSE_BAND_MASK=1 POSEDUP_ITEMS=pose_ok FRAMING_COND=0"; exit 1; }
for h in $(echo $COTRAIN_ALOE_HOSTS | tr , " "); do echo ",$PPM_HOSTS," | grep -q ",$h," || { echo "ABORT: aloe host $h not in PPM_HOSTS"; exit 1; }; done   # [raw]
[ -f "$POSEDUP_CLEAN_TARGET" ] || { echo "ABORT: [raw] clean-target report missing"; exit 1; }
[ "$ALOE_INIT_VOX" = 1 ] || { echo "ABORT: init from the PPMVOX export (ALOE_INIT_VOX=1)"; exit 1; }
[[ "$POSEDUP_PLAN_DIGEST" =~ ^[0-9a-f]{16}$ ]] || { echo "ABORT: POSEDUP_PLAN_DIGEST not set (tools/plan_posedup.py for QQA_STEPS $QQA_STEPS)"; exit 1; }   # [pdx]
for f in $COTRAIN_REF_CFG $ALOE_REF_CFG_26467 $ALOE_REF_CFG_28511 $ALOE_REF_CFG_29345 $ALOE_M1VOX_INDEX $ALOE_RAWGAIN $ALOE_SAMSEP_ROW_MAP $POSEDUP_POSE_INDEX; do
  [ -f $f ] || { echo "ABORT: missing $f"; exit 1; }; done
[ "$(sha256sum $POSEDUP_POSE_INDEX | cut -c1-64)" = "$POSEDUP_POSE_INDEX_SHA" ] || { echo "ABORT: POSE_INDEX.json sha256 != POSEDUP_POSE_INDEX_SHA"; exit 1; }   # [pdx]
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe.py $C/data_aloe_v2.py $C/data_aloe_v2vox.py $C/data_aloe_posedup.py $C/safety_ppm.py $C/optoff.py \
         $C/train_samsep_posedup.py $C/data_zoom.py $C/rowcache_vox.py $C/lib/audio_s4q.py $C/lib/voxtral_front.py $C/lib/vadapter_s4q.py; do
  $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
ALLOC=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort | paste -sd,)
[ "$ALLOC" = "$(echo $PPM_HOSTS | tr , '\n' | sort | paste -sd,)" ] || { echo "ABORT: allocation $ALLOC != PPM_HOSTS $PPM_HOSTS"; exit 1; }
echo ",$ALLOC," | grep -qE ",slinky-(9|10)," && { echo "ABORT: never slinky-9 / slinky-10 in this job"; exit 1; }
export PPM_HOSTS PPM_HOST_NRANKS PPM_HOST_GPUS
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR $P/logs; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== [raw] POSE-DUPLEX RAW CODES + ZOOM 1:1 (aloe hosts $COTRAIN_ALOE_HOSTS) job $SLURM_JOB_ID on $PPM_HOSTS (ranks $PPM_HOST_NRANKS; logical world $ALOE_LOGICAL_WORLD; cpus/task $SLURM_CPUS_PER_TASK) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN steps $QQA_STEPS resume '${COTRAIN_RESUME}'"
env | grep -E '^(QQA_|ALOE_|VOX_|COTRAIN_|POSEDUP_|POSE_BAND|FRAMING_|ZOOM_|PPM_HOST|S4_VOXTRAL|PPM_CD|PPM_DEC_ATTN|S3_MIN|HF_REPO|HFLX_)' | sort
FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ "$FD" -ge $((300 + 90)) ] && echo "/data free ${FD} GB: final consolidated /data export will be attempted (floor 300)" \
  || echo "NOTE: /data free ${FD} GB < 300 floor + 90: the final consolidated /data export will be REFUSED (node-local export-<last> + HF backup = final copy)"
# pre-flight on every host (one task per host): stores from /data, the A FLAC root symlink + the 8 ledgers of 29345's plan (every A track), the v5
# init subset staged for THIS host (STAGE.json host index == its PPM_HOSTS position) [pdx], a pose file readable [pdx], /scratch room, stale /dev/shm
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
H=$(hostname -s); cd /tmp
[ "${ALOE_STORES_FROM_DATA:-0}" = 1 ] && [ "$ALOE_SAMSEP_STORES" = "$ALOE_SAMSEP_SRC" ] || { echo "ABORT $H: stores must be read from /data (ALOE_STORES_FROM_DATA=1)"; exit 1; }
FR=$ALOE_FLAC_ROOT
if [ -L "$FR" ]; then [ "$(readlink "$FR")" = "$ALOE_SHARED_FLAC" ] || { echo "ABORT $H: $FR -> $(readlink "$FR") != $ALOE_SHARED_FLAC"; exit 1; }
elif [ -e "$FR" ]; then echo "ABORT $H: $FR exists as a real path (expected the symlink to the shared copy)"; exit 1
else mkdir -p "$(dirname "$FR")" && ln -s "$ALOE_SHARED_FLAC" "$FR" || { echo "ABORT $H: cannot link $FR"; exit 1; }; fi
for l in 0 1 2 3 4 5 6 7; do grep -q "\"plan_digest\": \"$QQA_REF_PLAN_DIGEST\"" "$FR/_ledger/DONE_v2vox_node$l.json" 2>/dev/null \
  || { echo "ABORT $H: logical node $l A tracks not verified for plan $QQA_REF_PLAN_DIGEST"; exit 1; }; done
I=$(echo "$PPM_HOSTS" | tr , "\n" | grep -nx "$H" | cut -d: -f1)
[ -f "$QQA_INIT/COMPLETE" ] && [ -f "$QQA_INIT/meta.json" ] && [ -f "$QQA_INIT/rep.pt" ] || { echo "ABORT $H: init $QQA_INIT not staged / COMPLETE on this host (stage_init_v5.sbatch)"; exit 1; }
grep -q "\"host_index\": \"$((I - 1))\"" "$QQA_INIT/STAGE.json" || { echo "ABORT $H: $QQA_INIT was staged for another host index (want $((I - 1))): restage in PPM_HOSTS order"; exit 1; }
'"$PY"' -I -c "import json,numpy as np,sys; j=json.load(open(sys.argv[1])); e=[s for r in j[\"rows\"][\"0\"].values() for s in r[\"sides\"].values() if s[\"ok\"]][0]; a=np.load(j[\"data_root\"]+\"/\"+e[\"path\"], mmap_mode=\"r\"); assert a.shape==(32,e[\"T\"])" "$POSEDUP_POSE_INDEX" \
  || { echo "ABORT $H: pose token files not readable"; exit 1; }
F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
NEED=$((ALOE_EXPORT_SCRATCH_FLOOR_GB + ${ALOE_PREFLIGHT_EXPORTS:-2} * 12 + ${ALOE_PREFLIGHT_SAVE_GB:-78}))   # [raw] main 10-29 23:4x: 1 intermediate export reserve (120 GB)   # main 10-25: the END FULL SAVE must never be skipped:
PS=/scratch/amartyavarma/ppmvox/ckpt/ppmvox-v5/dcp-18100   # 10-25: the v5 pause save is deleted (main approved) once the B merge is verified
if [ -d $PS ]; then CR=$(du -s --block-size=1G $PS | cut -f1); NEED=$((NEED - CR)); echo "$H: preflight credits ${CR} GB of the v5 pause save $PS (approved delete after the B merge verify, long before the end save)"; fi
[ "$F" -ge $NEED ] || { echo "ABORT $H: /scratch free ${F} GB < ${NEED} GB (floor $ALOE_EXPORT_SCRATCH_FLOOR_GB + 2 exports x 12 + end full save ${ALOE_PREFLIGHT_SAVE_GB:-78}); top dirs: $(du -s --block-size=1G /scratch/amartyavarma/* 2>/dev/null | sort -rn | head -6 | paste -sd" ")"; exit 1; }
mkdir -p "$ALOE_LOCAL_EXPORT_ROOT" "$COTRAIN_RESUME_ROOT" /scratch/amartyavarma/aloe-ft-32-samsep
if [ -n "$COTRAIN_RESUME" ]; then [ -f "$COTRAIN_RESUME/COMPLETE" ] || { echo "ABORT $H: resume state $COTRAIN_RESUME not COMPLETE on this host"; exit 1; }; fi
find /dev/shm -maxdepth 1 -user $USER \( -name "nccl-*" -o -name "torch_*" \) -mmin +5 -delete 2>/dev/null
G=$(echo "$PPM_HOST_GPUS" | cut -d";" -f$I)
echo "$H: stores /data, A tracks $(readlink $FR) ledgers ok, init $(cat $QQA_INIT/COMPLETE) ($(du -sh $QQA_INIT | cut -f1), host index $((I - 1))), pose ok; /scratch free ${F} GB; GPUs $G: $(nvidia-smi -i $G --query-gpu=index,temperature.gpu,clocks.sm,clocks.max.sm --format=csv,noheader | tr "\n" " ")"' || exit 1
# training: one srun step per host, rank = host offset + local id (code/task_samsep.sh)
PIDS=()
IFS=, read -ra HS <<< "$PPM_HOSTS"; IFS=, read -ra NR <<< "$PPM_HOST_NRANKS"
for i in "${!HS[@]}"; do
  srun -N1 -n${NR[$i]} -w ${HS[$i]} --ntasks-per-node=${NR[$i]} --kill-on-bad-exit=1 bash $C/task_samsep.sh $C/train_samsep_posedup.py &
  PIDS+=($!)
done
# HF watcher (poseduplex copy): one nice'd task per host, overlap step, no CPU binding, no GPU (private HF only, never GCS)
sleep 20
srun --overlap -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c 2 --mem=16G --gres=none --cpu-bind=none --kill-on-bad-exit=0 bash $P/hf_watch.sh > $P/logs/hf_watch-$SLURM_JOB_ID.log 2>&1 &
HFW=$!
alive() { local n=0 p; for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null && n=$((n + 1)); done; echo $n; }
while [ "$(alive)" = "${#PIDS[@]}" ]; do sleep 15; done
for i in $(seq 8); do [ "$(alive)" = 0 ] && break; sleep 15; done
[ "$(alive)" = 0 ] || { echo "== one host's step ended, the others still run after 120 s: stopping them $(TZ=America/Los_Angeles date '+%T PT')"; for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done; }
rc=0; for p in "${PIDS[@]}"; do wait $p; r=$?; [ $r -ne 0 ] && rc=$r; done
echo "== training exit $rc $(TZ=America/Los_Angeles date '+%T PT')"
echo "$rc" > $QQA_CONTROL_DIR/TRAIN_EXITED-$SLURM_JOB_ID
for i in $(seq 720); do kill -0 $HFW 2>/dev/null || break; sleep 15; done
kill -0 $HFW 2>/dev/null && { echo "== HF watcher still running after 3 h: stopping it (re-run hf_watch.sh to finish)"; kill $HFW; }
wait $HFW; echo "== hf watcher exit $? $(TZ=America/Los_Angeles date '+%T PT')"
exit $rc
