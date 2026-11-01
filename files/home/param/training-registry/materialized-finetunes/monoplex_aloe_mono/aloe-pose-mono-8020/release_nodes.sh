#!/bin/bash
# aloe-pose-mono-8020: runs after the training job ends (sbatch --dependency=afterany:<job>). Deletes ONLY this run's TTS drain file
# (si20k-tts/control/DRAIN_FOR_FT.posemono). Finished all 357 steps -> deleted at once. Otherwise (crash / stop) the file is kept up to 60 min
# so the owner can resume on the same nodes; if no new posemono8020 job is queued or running by then, it is deleted (no idle drained nodes).
# Usage: release_nodes.sh <job id>
J=$1; Q=/data/home/amartyavarma/ppm/aloe-pose-mono-8020; F=/data/home/amartyavarma/data/si20k-tts/control/DRAIN_FOR_FT.posemono
pt() { TZ=America/Los_Angeles date '+%F %T PT'; }
if grep -q "done 357 steps" $Q/logs/posemono8020-$J.log 2>/dev/null; then
  rm -f $F && echo "$(pt) job $J finished 357 steps: removed $F (TTS reclaims slinky-9..14,16,19)" | tee -a $Q/RELEASE.log; exit 0
fi
echo "$(pt) job $J ended WITHOUT finishing 357 steps: drain file kept up to 60 min for a resume" | tee -a $Q/RELEASE.log
for i in $(seq 1 120); do
  sleep 30
  squeue -h -u amartyavarma -n posemono8020 -o %i | grep -qv "^$J$" && { echo "$(pt) a new posemono8020 job exists: drain file kept (that job's release removes it)" | tee -a $Q/RELEASE.log; exit 0; }
done
rm -f $F && echo "$(pt) no resume within 60 min: removed $F" | tee -a $Q/RELEASE.log
