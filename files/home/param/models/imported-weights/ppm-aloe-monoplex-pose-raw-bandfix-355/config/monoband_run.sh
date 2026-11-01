#!/bin/bash
# pose6 MONOBAND (10-29, main -a8): = pose6/mono17850/mono17850_run.sh (job 31240) with ONLY: this kit's dirs / node-local tree
# /scratch/amartyavarma/pose6-monoband, code = pose6/code_bandmask, env monoband.env, the bandmask asserts (POSE_BAND_MASK=1, INPUT 0, FRAMING off)
# and the flag-on plan asserts (355 steps, rows digest ea070582180a59f0). mono17850_run.sh's header follows.
# pose6 B3 (10-27) mono17850 launch, exec'd by mono17850.sbatch (launch.sh submits it on --go only; nothing in this kit runs it by itself).
# = pose6/kit/ft_s2pg0_run.sh (= aloe_mono8020_run.sh + the base from FT_BASE, node rules, /scratch need + end-of-run write log, in-job preflight,
# rank-0 pin, compile caches in this kit's node-local tree) with: env mono17850.env, BOTH mods asserted OFF, preflight --mods none (base = the
# pre-mods key set exactly, no sidecar), this kit's dirs / port 29894 / node-local tree /scratch/amartyavarma/pose6-monoband, and the default
# save cadence = the recipe's hourly (FT_SAVES unset -> hourly). Stop: touch control/STOP_AFTER_SAVE. Never slinky-9.
set -uo pipefail
Q=/data/home/amartyavarma/ppm/pose6/monoband; KT=/data/home/amartyavarma/ppm/pose6/kit; C=/data/home/amartyavarma/ppm/pose6/code_bandmask; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
: "${FT_BASE:?FT_BASE (the base checkpoint dir) must be set: use launch.sh --base}"
FT_BASE=$(readlink -f "$FT_BASE")
export FT_NODES=$SLURM_NNODES
export FT_SAVES=${FT_SAVES:-hourly}   # [M17850] default = the recipe's hourly FULL saves (pose6/kit's default is final)
set -a; source ${QQA_ENV:-$Q/monoband.env}; set +a
. $KT/tools/ft_overrides.sh || { echo "ABORT: bad FT_LINK / FT_SAVES / FT_EXPORTS"; exit 1; }   # launch.sh user-call switches (--link-on, --saves, --final-export-only)
grep -q 'QQA_DECP_BB_OFF' $C/train_aloe_pose.py || { echo "ABORT: pose6/code does not read QQA_DECP_BB_OFF (the link knob would be silently ignored)"; exit 1; }
NEXP=$([ -z "$QQA_EXPORT_STEPS" ] && echo 1 || echo $(( $(echo $QQA_EXPORT_STEPS | tr , '\n' | grep -c .) + 1 )))
export MASTER_PORT=29894
# compile caches inside this kit's own node-local dir (ops only: same kernels; keeps the FT's node-local files in ONE tree for the hand-back)
export TRITON_CACHE_DIR=/scratch/amartyavarma/pose6-monoband/cache/triton TORCHINDUCTOR_CACHE_DIR=/scratch/amartyavarma/pose6-monoband/cache/inductor
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$QQA_POSE_B" = 1 ] && [ "$QQA_A_EMPTY" = 1 ] && [ "$QQA_POSE_HEAD_BUILD" = 0 ] || { echo "ABORT: needs QQA_POSE_B=1 QQA_A_EMPTY=1 QQA_POSE_HEAD_BUILD=0"; exit 1; }
[ "$ALOE_PERSONA_SIDES" = B ] && [ "$ALOE_VOICE_RULE" = p10 ] && [ "$ALOE_VOICE_P10" = 8 ] && [ "$ALOE_PERSONA_RULE" = p10 ] && [ "$ALOE_PERSONA_P10" = 2 ] && [ "$ALOE_DRAW_KEY" = item ] || { echo "ABORT: prompt rule env"; exit 1; }
[ "$ALOE_DATA" = aloe-mono ] && [ "$ALOE_STORE_DIR" = /data/home/amartyavarma/ppm/aloe-pose-ft/stores ] || { echo "ABORT: data source env"; exit 1; }
# [M17850] both mods OFF, no sidecar; mono8020's data / steps exactly (no Rosie data knobs); no test-only override
[ "${PPM_S2P:-0}" = 0 ] && [ "${PPM_G0COND:-0}" = 0 ] && [ -z "${ALOE_G0_SIDECAR:-}" ] || { echo "ABORT: mono17850 = both mods OFF (PPM_S2P / PPM_G0COND unset)"; exit 1; }
[ -z "${ALOE_TEST_NEWPARAM_RANDOM:-}" ] || { echo "ABORT: ALOE_TEST_NEWPARAM_RANDOM is test-only (never in a real run)"; exit 1; }
[ -z "${ALOE_MONO_ITEMS:-}" ] && [ "${ALOE_EPOCHS:-1}" = 1 ] && [ "$ALOE_ROWS_PER_STEP" = 64 ] && [ "$QQA_STEPS" = 355 ] && [ "$ALOE_ROWS_DIGEST" = ea070582180a59f0 ] || { echo "ABORT: data / steps must be the bandmask flag-on plan (355, ea070582180a59f0)"; exit 1; }
# [MONOBAND] the black-band fix: loss mask ON, input blanking OFF, framing OFF, masks + DROP_LANES present
[ "${POSE_BAND_MASK:-0}" = 1 ] && [ "${POSE_BAND_MASK_INPUT:-0}" = 0 ] && [ "${FRAMING_COND:-0}" = 0 ] && [ -f "$POSE_BAND_MASK_DIR/DROP_LANES.json" ] \
  && grep -q "POSE_BAND_MASK" $C/data_aloe_pose.py || { echo "ABORT: monoband = POSE_BAND_MASK=1, POSE_BAND_MASK_INPUT=0, FRAMING_COND=0, bandmask code + DROP_LANES"; exit 1; }
case "$SLURM_NNODES" in
  1) [ "${ALOE_OPT_OFFLOAD:-}" = 1 ] && [ "${ALOE_SKIP_FINAL_FULL:-}" = 1 ] && [ "$ALOE_SAVE_EVERY_MIN" = 100000 ] || { echo "ABORT: world 8 needs the [S2PG0-W8] overlay"; exit 1; } ;;
  8) [ -z "${ALOE_OPT_OFFLOAD:-}" ] && [ "${FT_SAVES:-hourly}" = hourly -a "$ALOE_SAVE_EVERY_MIN" = 60 -a -z "${ALOE_SKIP_FINAL_FULL:-}" -o "${FT_SAVES:-hourly}" != hourly ] \
       || { echo "ABORT: world 64 = mono8020's knobs (no world-8 overlay; FT_SAVES / FT_EXPORTS only via launch.sh --saves / --final-export-only)"; exit 1; } ;;
  *) echo "ABORT: 1 node (world 8) or 8 nodes (world 64 = mono8020's layout) only, got $SLURM_NNODES"; exit 1 ;;
esac
grep -q '@' ${QQA_ENV:-$Q/monoband.env} && { echo "ABORT: env still has @placeholders@"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe_pose.py $C/train_aloe_pose.py $C/framing.py $C/optoff.py $KT/tools/stage_base.py $KT/tools/preflight.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
# [S2PG0] rank 0 pinned to FT_RANK0 (launch.sh: slinky-15, the host with the most /scratch: rank 0 also writes rep.pt / meta.json of every export).
# Placement only (task_ppm.sh: rank = 8 x index in PPM_HOSTS + local id; each rank's data / seeds are unchanged); mono8020 used sort -V order.
if [ -n "${FT_RANK0:-}" ]; then
  echo ",$PPM_HOSTS," | grep -q ",$FT_RANK0," || { echo "ABORT: FT_RANK0 $FT_RANK0 not in $PPM_HOSTS"; exit 1; }
  export PPM_HOSTS=$(echo "$FT_RANK0,$(echo $PPM_HOSTS | tr , '\n' | grep -vx "$FT_RANK0" | paste -sd,)" | sed 's/,$//')
fi
case ",$PPM_HOSTS," in *,slinky-9,*) echo "ABORT: slinky-9 is never used"; exit 1;; esac
WORLD=$((8 * SLURM_NNODES)); R0=$(echo $PPM_HOSTS | cut -d, -f1)
# ---- the base: local-shards = read in place by load_init (mono8020's path: it read local-11488 so, from /data); consolidated DCP (a 31101 w-N)
#      = read directly by pose6/code's load_init_dcp (default; DCP reshards in place, bf16 -> fp32 copy) or, with FT_STAGE=1, staged node-local
#      into local-shards first (tools/stage_base.py, bit-exact) so mono8020's own load_init loop reads it (+90 GB /scratch / node count)
if [ -f "$FT_BASE/meta.json" ]; then
  export QQA_INIT=$FT_BASE; STAGE=0; STG_GB=0
elif [ -f "$FT_BASE/.metadata" ] && [ -f "$FT_BASE/COMPLETE" ] && [ "${FT_STAGE:-0}" = 1 ]; then
  export QQA_INIT=/scratch/amartyavarma/pose6-monoband/base-$(basename $FT_BASE)-W$WORLD; STAGE=1; STG_GB=$((90 / SLURM_NNODES + 4))
elif [ -f "$FT_BASE/.metadata" ] && [ -f "$FT_BASE/COMPLETE" ]; then
  grep -q 'def load_init_dcp' $C/train_aloe_pose.py || { echo "ABORT: pose6/code has no DCP init path: use FT_STAGE=1"; exit 1; }
  export QQA_INIT=$FT_BASE; STAGE=0; STG_GB=0
else
  echo "ABORT: FT_BASE $FT_BASE is neither a COMPLETE consolidated DCP nor a ppm-local-bf16-shards-v1 dir"; exit 1
fi
# one run tag for the whole run (first job's id), reused on every resume so metrics.jsonl stays one file: RUN_TAG.txt (delete it + the saves
# to start a fresh run)
if [ -f $Q/RUN_TAG.txt ]; then export QQA_RUN=$(cat $Q/RUN_TAG.txt) FRESH=0; else export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID FRESH=1; fi   # written once every check passed (below)
mkdir -p $QQA_CONTROL_DIR
echo "job $SLURM_JOB_ID nodes $PPM_HOSTS started $(TZ=America/Los_Angeles date '+%F %T PT') run $QQA_RUN base $FT_BASE init $QQA_INIT" > $Q/RUNNING_JOB.txt
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== pose6 monoband (mono17850 recipe + POSE_BAND_MASK=1, init dcp-17850) job $SLURM_JOB_ID on $PPM_HOSTS (world $WORLD) $(TZ=America/Los_Angeles date '+%F %T PT') base $FT_BASE init $QQA_INIT run $QQA_RUN"
env | grep -E '^(QQA_|ALOE_|PPM_CD|PPM_DEC_ATTN|PPM_S2P|PPM_G0COND|S3_MIN|FT_|POSE_BAND|FRAMING)' | sort
echo "== D33 decp_bb link: $([ $QQA_DECP_BB_OFF = 1 ] && echo 'OFF (= base 31101; frozen, never computed)' || echo 'ON (= mono8020 knob value; --link-on)')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
# ---- node checks: /scratch need (world 64: mono8020's 1500 / nodes + 50 GB + the staged base slice; world 8: staged base 90 + 6 node-local
#      exports x 83 GiB (88.4 GB) + 30, no FULL saves; all in df -BG units), our 8 GPUs idle + not thermally throttled, stores readable
# ---- /scratch need per host (df -BG units = GiB). DEFAULT rule 'floor' (main 10-27): the FT's measured write + DISK_FLOOR_GIB (= pose5's floor for
#      31101). Write per host at 64 ranks (24272's [build] params 44.176B, decoders 1.150B replicated; export 88.4 GB): FULL save = 43.03B x 12 B / 8
#      = 60.1 + node.pt decoders (fp32 + AdamW) 12.9 = 73 GiB; export slice = 10.0 GiB (+ rep.pt 2.2 on rank 0) -> saves final 135 (rank 0 148),
#      hourly (2 kept) 208 (221), none 62 (75). Rule 'recipe' = mono8020's 1500 / 1000 / 500 per 8 nodes + 50 (no rank-0 extra).
DISK_FLOOR_GIB=30
case "${FT_SAVES:-hourly}" in hourly) NSAVE=2;; final) NSAVE=1;; none) NSAVE=0;; esac
# write per host (GiB, x10 integer maths): NSAVE x 73 + NEXP x 10.3, rank 0 + NEXP x 2.2 (rep.pt); recipe rule = (NSAVE x 500 + NEXP x 83) / 8 + 50
WR=$(( (NSAVE * 730 + NEXP * 103 + 9) / 10 )); R0X=$(( (NEXP * 22 + 9) / 10 )); SV=$(( NSAVE * 500 + NEXP * 83 ))
if [ "$SLURM_NNODES" = 8 ]; then
  if [ "${FT_DISK_RULE:-floor}" = recipe ]; then NEED=$((SV / SLURM_NNODES + 50 + STG_GB)); NEED0=$NEED; else NEED=$((WR + DISK_FLOOR_GIB + STG_GB)); NEED0=$((NEED + R0X)); fi
else NEED=$((STG_GB + NEXP * 83 + 30)); NEED0=$NEED; fi   # fallback (world 8): exports 83 GiB each + 30
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=2 --mem=4G --cpu-bind=none bash -c '
H=$(hostname -s); mkdir -p /scratch/amartyavarma/pose6-monoband
fr=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); used=$(du -sBG /scratch/amartyavarma/pose6-monoband 2>/dev/null | cut -f1 | tr -dc 0-9)
N_=$([ "$H" = '$R0' ] && echo '$NEED0' || echo '$NEED')
echo "$H /scratch $fr GiB free (+ ${used:-0} GiB already in pose6-monoband), need $N_ GiB (rule '${FT_DISK_RULE:-floor}', saves '${FT_SAVES:-final}'$([ "$H" = '$R0' ] && echo ", rank 0"))"; [ $((fr + ${used:-0})) -ge $N_ ] || { echo "$H ABORT: /scratch $fr GiB free < NEEDED $N_ GiB"; exit 1; }
for g in 0 1 2 3 4 5 6 7; do q=$(nvidia-smi -i $g --query-gpu=index,memory.used,clocks.sm,clocks.max.sm,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader,nounits)
  echo "$H gpu $q"; [ "$(echo "$q" | cut -d, -f2 | tr -dc 0-9)" -lt 1000 ] || { echo "$H ABORT: GPU $g busy: $q"; exit 1; }
  echo "$q" | cut -d, -f5,6 | tr -d " " | grep -qE "(^|,)Active(,|$)" && { echo "$H ABORT: GPU $g thermal slowdown: $q"; exit 1; }; done
if [ "$FRESH" = 1 ] && ls -d $ALOE_SAVE_ROOT/step-* > /dev/null 2>&1; then echo "$H ABORT: fresh run but stale saves in $ALOE_SAVE_ROOT (ALOE_RESUME=auto would resume them)"; exit 1; fi
ls /data/home/amartyavarma/ppm/aloe-pose-ft/stores/STORES.json > /dev/null || exit 1' || { echo "ABORT: node checks failed"; exit 1; }
# ---- [M17850] base preflight (key set = the FT model's exactly, no s_emb / g0, bf16)
srun -N1 -n1 -w $R0 --cpus-per-task=4 --mem=32G --cpu-bind=none $PY -u $KT/tools/preflight.py --mods none --base $FT_BASE --code $C --json $Q/logs/preflight-$SLURM_JOB_ID.json \
  || { echo "ABORT: preflight FAILED (base keys / shapes): see above"; exit 1; }
# ---- [S2PG0] stage the DCP base node-local (one task per node, each writes its own ranks' files + rank0 / rep / meta; bit-exact re-read check)
if [ $STAGE = 1 ]; then
  srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=8 --mem=200G --cpu-bind=none $PY -u $KT/tools/stage_base.py \
    --src $FT_BASE --out $QQA_INIT --world $WORLD --hosts $PPM_HOSTS || { echo "ABORT: base staging failed"; exit 1; }
  echo "== base staged node-local $(TZ=America/Los_Angeles date '+%T PT'): $FT_BASE -> $QQA_INIT on $PPM_HOSTS"
fi
[ "$FRESH" = 1 ] && echo $QQA_RUN > $Q/RUN_TAG.txt
rc=0
while :; do
  rm -f "${QQA_CONTROL_DIR:?}/STOP_AFTER_SAVE" "${QQA_CONTROL_DIR:?}/STOPPED"
  # (--mem=0 inside the job = every step gets the job's whole per-node memory = mono8020's srun line without --mem; written out per the --mem rule)
  srun --ntasks-per-node=8 --cpus-per-task=${ALOE_CPT:-4} --mem=0 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_pose.py &
  SRUN=$!
  while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
  echo "== trainer exit $rc $(TZ=America/Los_Angeles date '+%T PT')"
  if [ $rc = 0 ] && [ -f $QQA_CONTROL_DIR/STOPPED ] && [ -f $QQA_CONTROL_DIR/RESTART_AFTER_STOP ]; then
    rm -f "${QQA_CONTROL_DIR:?}/RESTART_AFTER_STOP"
    echo "== stopped at step $(cat $QQA_CONTROL_DIR/STOPPED) with RESTART_AFTER_STOP: in-job resume (ALOE_RESUME=$ALOE_RESUME) $(TZ=America/Los_Angeles date '+%T PT')"
    continue
  fi
  break
done
# ---- [S2PG0] the FT's ACTUAL /scratch write per host (confirms the measured-write rule; read only)
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=1 --mem=1G --cpu-bind=none bash -c '
H=$(hostname -s); d=/scratch/amartyavarma/pose6-monoband; g() { du -sBG "$1" 2>/dev/null | cut -f1 | tr -dc 0-9; }
echo "[disk] $H wrote exports $(g $d/exports) saves $(g $d/saves) cache $(g $d/cache) base $(ls -d $d/base-* > /dev/null 2>&1 && du -csBG $d/base-* | tail -1 | cut -f1 | tr -dc 0-9 || echo 0) total $(g $d) GiB; /scratch free now $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9) GiB (need was '$NEED' / rank 0 '$NEED0')"' 2>&1 | tee $Q/logs/disk-$SLURM_JOB_ID.txt
echo "job $SLURM_JOB_ID nodes $PPM_HOSTS ended $(TZ=America/Los_Angeles date '+%F %T PT') rc $rc" >> $Q/RUNNING_JOB.txt
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
