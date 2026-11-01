#!/bin/bash
# poseduplex launcher (login node: sbatch only). 8 hosts x 8 GPUs = world 64 (rank = 8 h + local; host order = the staging order).
# usage: bash launch_poseduplex.sh h0 h1 h2 h3 h4 h5 h6 h7 [dependency]
#   env knobs passed through: QQA_STEPS_SET (default 498, must have its plan lines in run_poseduplex.env), POSEDUP_INIT_TAG (default: control/INIT_TAG
#   written by stage_init_v5.sbatch), COTRAIN_RESUME (resume dir, same hosts / order).
set -euo pipefail
P=/data/home/amartyavarma/ppm/poseduplex-raw; EF=$P/run_poseduplex.env
[ $# -ge 8 ] || { echo "usage: bash launch_poseduplex.sh h0 h1 h2 h3 h4 h5 h6 h7 [dependency]"; exit 1; }
HOSTS=$(printf '%s,' "${@:1:8}" | sed 's/,$//'); DEP=${9:-}
[ "$(echo $HOSTS | tr , '\n' | sort -u | wc -l)" = 8 ] || { echo "ABORT: 8 distinct hosts"; exit 1; }
echo ",$HOSTS," | grep -qE ',slinky-(9|10),' && { echo "ABORT: never slinky-9 / slinky-10"; exit 1; }
TAG=${POSEDUP_INIT_TAG:-$(cat $P/control/INIT_TAG 2>/dev/null || true)}
[[ "$TAG" =~ ^w-[0-9]+$ ]] || { echo "ABORT: init tag unknown (run stage_init_v5.sbatch first, or set POSEDUP_INIT_TAG=w-<N>)"; exit 1; }
squeue -h -u $USER -o '%j' | grep -qx pdxraw && { echo "ABORT: a pdxraw job is already queued / running"; exit 1; }
STEPS=${QQA_STEPS_SET:-424}
DG=$(POSEDUP_HOSTS=$HOSTS QQA_STEPS_SET=$STEPS bash -c "set -a; source $EF; echo \$POSEDUP_PLAN_DIGEST")
[[ "$DG" =~ ^[0-9a-f]{16}$ ]] || { echo "ABORT: no plan-table line for QQA_STEPS_SET=$STEPS in run_poseduplex.env ($DG): tests/run_tests.sh plan $STEPS"; exit 1; }
CLEAN=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM|QQA|COTRAIN|ALOE|VOX|ZOOM|HF_REPO|HFLX|HFW|POSEDUP)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' || true)
J=$(env $CLEAN QQA_ENV=$EF POSEDUP_HOSTS=$HOSTS POSEDUP_INIT_TAG=$TAG QQA_STEPS_SET=$STEPS COTRAIN_RESUME=${COTRAIN_RESUME:-} \
    sbatch --parsable ${DEP:+--dependency=$DEP} -N8 -w $HOSTS --ntasks-per-node=8 --cpus-per-task=6 --mem=520G $P/run_poseduplex.sbatch)
printf '{"job": "%s", "hosts": "%s (8 x 8 GPUs, world 64, this order)", "init_tag": "%s", "steps": %s, "env": "%s", "resume": "%s", "dependency": "%s", "time": "%s"}\n' \
  "$J" "$HOSTS" "$TAG" "$STEPS" "$EF" "${COTRAIN_RESUME:-}" "$DEP" "$(TZ=America/Los_Angeles date '+%F %T PT')" > $P/LAUNCHED_JOB
cat $P/LAUNCHED_JOB
