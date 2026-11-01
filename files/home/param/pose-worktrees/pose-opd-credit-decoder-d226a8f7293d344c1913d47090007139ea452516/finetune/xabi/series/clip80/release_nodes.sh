#!/bin/bash
# aloe-v2pure-clip80: runs after the training job ends (sbatch --dependency=afterany:<job>). Deletes ONLY this run's TTS drain file
# (si20k-tts/control/DRAIN_FOR_FT.monoplex) and only if the trainer finished all 200 steps; otherwise keeps the nodes drained (crash /
# stop -> the owner decides). Usage: release_nodes.sh <job id>
J=$1; Q=/data/home/amartyavarma/ppm/aloe-v2pure-clip80; F=/data/home/amartyavarma/data/si20k-tts/control/DRAIN_FOR_FT.monoplex
pt() { TZ=America/Los_Angeles date '+%F %T PT'; }
if grep -q "done 200 steps" $Q/logs/aloeclip80-$J.log 2>/dev/null; then
  rm -f $F && echo "$(pt) job $J finished 200 steps: removed $F (TTS reclaims slinky-0,1,2)" | tee -a $Q/RELEASE.log
else
  echo "$(pt) job $J ended WITHOUT finishing 200 steps: drain file kept" | tee -a $Q/RELEASE.log
fi
