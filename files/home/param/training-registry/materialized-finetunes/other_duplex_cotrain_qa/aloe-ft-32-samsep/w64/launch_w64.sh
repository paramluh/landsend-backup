#!/bin/bash
# w64 launcher (login node: sbatch only): 8 hosts x 8 GPUs, run_w64.sbatch on w64/run_w64.env. usage: bash launch_w64.sh [dependency]
set -euo pipefail
Q=/data/home/amartyavarma/ppm/aloe-ft-32-samsep; EF=$Q/w64/run_w64.env; DEP=${1:-}
HOSTS=$(grep -E '^PPM_HOSTS=' $EF | cut -d= -f2 | cut -d' ' -f1)
squeue -h -u $USER -o '%j' | grep -qx aloesamsep && { echo "ABORT: an aloesamsep job is already queued / running"; exit 1; }
CLEAN=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM|QQA|COTRAIN|ALOE|VOX|ZOOM|HF_REPO|HFLX|HFW)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' || true)
J=$(env $CLEAN QQA_ENV=$EF COTRAIN_RESUME= sbatch --parsable ${DEP:+--dependency=$DEP} -N8 -w $HOSTS \
    --ntasks-per-node=8 --cpus-per-task=6 --mem=520G $Q/w64/run_w64.sbatch)
printf '{"job": "%s", "hosts": "%s (8 x 8 GPUs, world 64)", "env": "%s", "dependency": "%s", "time": "%s"}\n' \
  "$J" "$HOSTS" "$EF" "$DEP" "$(TZ=America/Los_Angeles date '+%F %T PT')" > $Q/w64/LAUNCHED_JOB
cat $Q/w64/LAUNCHED_JOB
