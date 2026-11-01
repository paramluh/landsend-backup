#!/bin/bash
#SBATCH -p all -N1 -n1 -c 1 --mem=1G --gres=none -t 3:00:00 -J si20k11488-drop --exclude=slinky-4,slinky-18
#SBATCH -o /data/home/amartyavarma/ppm/si20k-vox-ppm11488/logs/drop-%j.out
# coordinator 10-19: once export-165 is HF-verified (BACKUP_LOG 'export-165 HF upload rc 0'), delete the node-local export-132 / 99 / 66 of
# si20k-vox-ppm11488 on the 8 PPM hosts (not exported anywhere: the run's result is recorded, these intermediates are not needed; /data has no
# room for them). Logged per host. If the export-165 upload fails, nothing is deleted.
Q=/data/home/amartyavarma/ppm/si20k-vox-ppm11488; BL=$Q/exports/BACKUP_LOG.txt; LOC=/scratch/amartyavarma/si20k-vox-ppm11488/exports
HP=slinky-0,slinky-1,slinky-3,slinky-5,slinky-8,slinky-9,slinky-16,slinky-17
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
while :; do
  grep -q "export-165 HF upload rc 0" $BL && break
  grep -q "export-165 HF upload rc [1-9]" $BL && { echo "[$(pt)] export-165 HF upload FAILED: nothing deleted" | tee -a $BL; exit 1; }
  sleep 20
done
out=$(env -i PATH=/usr/bin:/bin HOME=$HOME USER=$USER srun -p all -w $HP -N8 -n8 --ntasks-per-node=1 -c1 --mem=1G --gres=none -t 10 -J si20k11488-rm bash -c "source /data/home/amartyavarma/ppm/scratch_guard.sh; for N in 132 99 66; do [ -d $LOC/export-\$N ] && guard_rm $LOC/export-\$N; done; echo \"RM \$(hostname -s) \$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)G left: \$(ls $LOC 2>/dev/null | tr '\n' ' ')\"" 2>&1 | grep '^RM' | sort -V | tr '\n' ';')
echo "[$(pt)] export-165 HF-verified -> node-local export-132 / 99 / 66 DELETED (not exported; coordinator 10-19): $out" | tee -a $BL >> $Q/logs/CLEANUP.log
