#!/bin/bash
# aloe speaker LoRA FT launcher (login node: one CPU srun for the plan check + sbatch only). NOT RUN during prep (10-24): main / the user decide.
#   1. rebuilds / verifies the plan for the env's knobs on a CPU node (lora/tools/build_spk.py) and refuses when a derived env line differs
#      (QQA_WARMUP / QQA_EXPORT_STEPS are knobs: not compared);
#   2. sbatch -N1 on slinky-18 (8 GPUs, 8 tasks x 6 CPUs = 48 of its 54 free CPUs, --mem 400G: the pinned AdamW offload ~40 GB + workers).
# usage: bash launch_brianna.sh [dependency]
set -euo pipefail
Q=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora; EF=${QQA_ENV:-$Q/rosie/run_rosie.env}; DEP=${1:-}
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
get() { grep -E "^$1=" $EF | head -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*$//' | tr -d '"' | awk '{print $1}'; }
ACC=$(get ALOE_SPK_ACCOUNT); LW=$(get ALOE_LOGICAL_WORLD); EP=$(get ALOE_SPK_EPOCHS); ST=$(get ALOE_SPK_STEPS); EFR=$(get ALOE_SPK_EVAL_FRAC)
EC=$(get ALOE_SPK_EVAL_CALLS); RF=$(get ALOE_REPLAY_FRAC); HOSTS=$(get PPM_HOSTS)
EVERY=$(get QQA_EXPORT_STEPS | cut -d, -f1); EVERY=${EVERY:-10}
[ -n "$DEP" ] || { squeue -h -u $USER -o "%j" | grep -qx aloespklora && { echo "ABORT: an aloespklora job is already queued / running"; exit 1; }; }
OUT=$(timeout 900 srun -p all -w slinky-4 -N1 -n1 -c16 --mem=32G -t 15 -J spk-build $PY -I $Q/tools/build_spk.py $ACC $LW $EP $ST $EFR $EC $RF $EVERY 2>/dev/null | grep -E '^[A-Z_0-9]+=' || true)
[ -n "$OUT" ] || { echo "ABORT: lora/tools/build_spk.py failed"; exit 1; }
bad=0
for kv in $OUT; do k=${kv%%=*}; v=${kv#*=}
  case $k in QQA_WARMUP|QQA_EXPORT_STEPS) continue;; esac
  [ "$(get $k)" = "$v" ] || { echo "ENV MISMATCH $k: env '$(get $k)' plan '$v'"; bad=1; }; done
[ $bad = 0 ] || { echo "ABORT: put the plan's lines above into $EF"; exit 1; }
CLEAN=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM|QQA|COTRAIN|ALOE|VOX|ZOOM|HF_REPO|HFLX|HFW)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' || true)
J=$(env $CLEAN QQA_ENV=$EF COTRAIN_RESUME= sbatch --parsable ${DEP:+--dependency=$DEP} -N1 -w $HOSTS \
    --ntasks-per-node=6 --cpus-per-task=6 --mem=400G $Q/run_brianna.sbatch)
printf '{"job": "%s", "hosts": "%s (GPUs 0-5, world 6)", "account": "%s", "env": "%s", "dependency": "%s", "time": "%s"}\n' \
  "$J" "$HOSTS" "$ACC" "$EF" "$DEP" "$(TZ=America/Los_Angeles date '+%F %T PT')" > $Q/rosie/LAUNCHED_JOB
cat $Q/LAUNCHED_JOB
