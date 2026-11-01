#!/bin/bash
# aloe speaker FT launcher (login node: one CPU srun for the plan check + sbatch only). NOT RUN during prep (10-24): the user decides.
#   1. rebuilds / verifies the speaker plan for the env's knobs on a CPU node (tools/build_spk.py: plan file, RAWGAIN, A-FLAC ledgers) and refuses
#      when a derived env line (digest, steps, counts, files) differs from the plan's (QQA_WARMUP / QQA_EXPORT_STEPS are the user's knobs: not compared);
#   2. sbatch -N2 on PPM_HOSTS (2 x 8 GPUs), 29345's per-task resources (8 tasks x 6 CPUs per node; --mem 600G: + the pinned AdamW offload).
# usage: bash launch_speaker.sh [dependency]
set -euo pipefail
Q=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker; EF=${QQA_ENV:-$Q/run_speaker.env}; DEP=${1:-}
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
get() { grep -E "^$1=" $EF | head -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*$//' | tr -d '"' | awk '{print $1}'; }
ACC=$(get ALOE_SPK_ACCOUNT); LW=$(get ALOE_LOGICAL_WORLD); EP=$(get ALOE_SPK_EPOCHS); ST=$(get ALOE_SPK_STEPS); EFR=$(get ALOE_SPK_EVAL_FRAC)
HOSTS=$(get PPM_HOSTS)
squeue -h -u $USER -o '%j' | grep -qx aloespk && { echo "ABORT: an aloespk job is already queued / running"; exit 1; }
OUT=$(timeout 900 srun -p all -w slinky-18 -N1 -n1 -c8 --mem=32G -t 15 -J spk-build $PY -I $Q/tools/build_spk.py $ACC $LW $EP $ST $EFR 2>/dev/null | grep -E '^[A-Z_0-9]+=' || true)
[ -n "$OUT" ] || { echo "ABORT: tools/build_spk.py failed"; exit 1; }
bad=0
for kv in $OUT; do k=${kv%%=*}; v=${kv#*=}
  case $k in QQA_WARMUP|QQA_EXPORT_STEPS) continue;; esac
  [ "$(get $k)" = "$v" ] || { echo "ENV MISMATCH $k: env '$(get $k)' plan '$v'"; bad=1; }; done
[ $bad = 0 ] || { echo "ABORT: put the plan's lines above into $EF"; exit 1; }
[ "$LW" = 16 ] || [ "$(get ALOE_SPK_DEC_PER_SLOT)" = 1 ] || [ "$(get QQA_DEC_K)" = 1 ] || { echo "ABORT: logical world $LW on 16 GPUs with QQA_DEC_K > 1 needs ALOE_SPK_DEC_PER_SLOT=1"; exit 1; }
CLEAN=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM|QQA|COTRAIN|ALOE|VOX|ZOOM|HF_REPO|HFLX|HFW)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' || true)
J=$(env $CLEAN QQA_ENV=$EF COTRAIN_RESUME= sbatch --parsable ${DEP:+--dependency=$DEP} -N2 -w $HOSTS \
    --ntasks-per-node=8 --cpus-per-task=6 --mem=600G $Q/run_speaker.sbatch)
printf '{"job": "%s", "hosts": "%s (2 x 8 GPUs, world 16)", "account": "%s", "logical_world": "%s", "env": "%s", "dependency": "%s", "time": "%s"}\n' \
  "$J" "$HOSTS" "$ACC" "$LW" "$EF" "$DEP" "$(TZ=America/Los_Angeles date '+%F %T PT')" > $Q/LAUNCHED_JOB
cat $Q/LAUNCHED_JOB
