#!/bin/bash
# aloe-ft-32-samsep launcher (login node: checks + sbatch only). usage: bash launch.sh [init path] [dependency]
#   init path (optional) = override run.env's QQA_INIT for this launch only (the one-line base swap; default: run.env's line).
#   resume: COTRAIN_RESUME=/scratch/amartyavarma/aloe-ft-32-samsep/saves/<run>-j<job>/dcp-<N> bash launch.sh
set -euo pipefail
Q=/data/home/amartyavarma/ppm/aloe-ft-32-samsep
INIT=${1:-$(grep -E '^QQA_INIT=' $Q/run.env | cut -d= -f2)}
DEP=${2:-}
[ -n "$INIT" ] || { echo "ABORT: no QQA_INIT"; exit 1; }
# a launch-only override is written into a copy of run.env next to it (the run's env file is then the exact record of the launch)
EF=$Q/run.env
if [ "$INIT" != "$(grep -E '^QQA_INIT=' $Q/run.env | cut -d= -f2)" ]; then
  EF=$Q/run.env.launch-$(date +%Y%m%d-%H%M%S); sed "s#^QQA_INIT=.*#QQA_INIT=$INIT#" $Q/run.env > $EF; fi
squeue -h -u $USER -o '%j' | grep -qx aloesamsep && { echo "ABORT: an aloesamsep job is already queued / running"; exit 1; }
CLEAN=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM|QQA|COTRAIN|ALOE|VOX|ZOOM|HF_REPO|HFLX)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' || true)
J=$(env $CLEAN QQA_ENV=$EF COTRAIN_RESUME=${COTRAIN_RESUME:-} sbatch --parsable ${DEP:+--dependency=$DEP} -N2 -w slinky-4,slinky-18 \
    --ntasks-per-node=8 --cpus-per-task=4 --mem=520G $Q/run.sbatch)
printf '{"job": "%s", "hosts": "slinky-18 (8 ranks) + slinky-4 (GPUs 0-5, 6 ranks)", "init": "%s", "env": "%s", "resume": "%s", "dependency": "%s", "time": "%s"}\n' \
  "$J" "$INIT" "$EF" "${COTRAIN_RESUME:-}" "$DEP" "$(TZ=America/Los_Angeles date '+%F %T PT')" > $Q/LAUNCHED_JOB
cat $Q/LAUNCHED_JOB
echo "log: $Q/logs/aloesamsep-$J.log   (HF watcher: $Q/logs/hf_watch-$J-<host>.log, BACKUP_LOG.txt)"
echo "gather the final export for the demo format after the run: sbatch --dependency=afterok:$J -N2 -w slinky-4,slinky-18 $Q/gather_final.sbatch"
