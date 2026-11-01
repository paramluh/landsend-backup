#!/bin/bash
#SBATCH -p all -N1 -n1 -c 1 --mem=1G --gres=none -t 12:00:00 -J si20knoc-chain --exclude=slinky-4,slinky-18
#SBATCH -o /data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt/logs/chain-%j.out
# si20k-vox-ppm-nocorrupt CHAINED LAUNCH (launch only): submits the no-corruption run + its watcher on the 8 PPM hosts once si20k-vox-ppm11488's
# FINAL export has landed AND is HF-verified (FINAL_EXPORT_READY + export-198/HF_BACKUP.json). FALLBACKS (never leave the nodes idle): the 11488
# job ended without a final (FT_ENDED_NO_FINAL) -> launch now; the 11488 job has been gone for 20 min and the HF verify is still missing ->
# launch anyway (logged). The drain file DRAIN_FOR_FT.si20kvoxnocorrupt (written at prep) keeps the TTS off the hosts in between.
M=/data/home/amartyavarma/ppm/si20k-vox-ppm11488; Q=/data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt
HP=slinky-0,slinky-1,slinky-3,slinky-5,slinky-8,slinky-9,slinky-16,slinky-17
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
JM=$(cat $M/LAUNCHED_JOB); gone=""
echo "[$(pt)] chain job $SLURM_JOB_ID: waiting for $M (job $JM) final export + HF verify"
while :; do
  [ -f $Q/LAUNCH_DONE ] && { echo "[$(pt)] already launched: $(cat $Q/LAUNCH_DONE)"; exit 0; }
  why=""
  if [ -f $M/FINAL_EXPORT_READY ] && [ -f $M/export-198/HF_BACKUP.json ]; then why="11488 final export landed + HF-verified"
  elif [ -f $M/FT_ENDED_NO_FINAL ]; then why="FALLBACK: 11488 ended without a final export"
  elif [ -z "$(squeue -h -j $JM 2>/dev/null)" ]; then
    [ -z "$gone" ] && gone=$(date +%s)
    [ $(( $(date +%s) - gone )) -gt 1200 ] && why="FALLBACK: 11488 job gone 20 min, HF verify still missing (final $(cat $M/FINAL_EXPORT_READY 2>/dev/null))"
  fi
  if [ -n "$why" ]; then
    j=$(sbatch --parsable -w $HP $Q/si20k_vox_ppm_nocorrupt.sbatch) || { echo "[$(pt)] sbatch failed; retry"; sleep 30; continue; }
    w=$(sbatch --parsable --exclude=slinky-4,slinky-18 -o $Q/logs/watch-%j.out $Q/tools/watch_noc.sbatch $Q $j si20kvoxnocorrupt 80 600 -)
    echo $j > $Q/LAUNCHED_JOB; echo "$(pt) nocorrupt $j (watch $w): $why" | tee $Q/LAUNCH_DONE
    echo "- $(pt): launched job $j (chain: $why)." >> $Q/README.md
    exit 0
  fi
  sleep 30
done
