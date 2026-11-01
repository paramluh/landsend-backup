#!/bin/bash
# PERSONA launcher (login node: sbatch only) = w64/launch_w64.sh for persona/run_persona.env + persona/run_persona.sbatch (8 hosts x 8 GPUs,
# world 64). usage: bash /data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/launch_persona.sh [dependency]
set -euo pipefail
Q=/data/home/amartyavarma/ppm/aloe-ft-32-samsep; EF=$Q/persona/run_persona.env; DEP=${1:-}
HOSTS=$(grep -E '^PPM_HOSTS=' $EF | cut -d= -f2 | cut -d' ' -f1)
squeue -h -u $USER -o '%j' | grep -qx aloepersona && { echo "ABORT: an aloepersona job is already queued / running"; exit 1; }
CLEAN=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM|QQA|COTRAIN|ALOE|VOX|ZOOM|HF_REPO|HFLX|HFW)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' || true)
J=$(env $CLEAN QQA_ENV=$EF COTRAIN_RESUME= sbatch --parsable ${DEP:+--dependency=$DEP} -N8 -w $HOSTS \
    --ntasks-per-node=8 --cpus-per-task=6 --mem=520G $Q/persona/run_persona.sbatch)
printf '{"job": "%s", "hosts": "%s (8 x 8 GPUs, world 64)", "env": "%s", "dependency": "%s", "time": "%s"}\n' \
  "$J" "$HOSTS" "$EF" "$DEP" "$(TZ=America/Los_Angeles date '+%F %T PT')" > $Q/persona/LAUNCHED_JOB
cat $Q/persona/LAUNCHED_JOB
echo "log: $Q/persona/logs/aloepersona-$J.log"
