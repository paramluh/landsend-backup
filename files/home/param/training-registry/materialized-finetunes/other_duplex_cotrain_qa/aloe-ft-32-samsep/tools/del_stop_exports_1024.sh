#!/bin/bash
# main (amartyavarma-48) go 10-24 ~00:00 PT: free space after FT 28249 stop. Node-local export-83 / export-stop90 (HF VERIFY PASS both hosts) + on slinky-18 the /data copy export-249-stop90.
source /data/home/amartyavarma/ppm/scratch_guard.sh
H=$(hostname -s); LOG=/data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
E=/scratch/amartyavarma/aloe-ft-32-samsep/exports
b=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
for t in export-83:InternalCan/ppm-aloe-samsep-16861 export-stop90:InternalCan/ppm-aloe-samsep-16861-final; do
  d=$E/${t%%:*}; r=${t#*:}
  [ -e $d ] || { echo "$H $d absent (already deleted)"; continue; }
  sz=$(du -sh $d | cut -f1)
  if guard_rm $d; then echo "[$(pt)] aloe-ft-32-samsep (ALOEFT32, main amartyavarma-48 go): deleted $H:$d ($sz, node-local bf16 export) after HF VERIFY PASS -> private $r/${t%%:*}/" | tee -a $LOG
  else echo "$H $d guard_rm REFUSED"; fi
done
if [ $H = slinky-18 ]; then
  d=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/export-249-stop90
  if [ -e $d ]; then sz=$(du -sh $d | cut -f1)
    guard_rm $d && echo "[$(pt)] aloe-ft-32-samsep (ALOEFT32, main amartyavarma-48 go): deleted $d ($sz, /data consolidated stop export; weights on private HF InternalCan/ppm-aloe-samsep-16861-final/export-stop90/ VERIFY PASS) /data near 300 GB floor" | tee -a $LOG || echo "$d guard_rm REFUSED"
  fi
  echo "/data free now $(df -BG --output=avail /data | tail -1 | tr -d ' ')"
fi
a=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
echo "$H /scratch free ${b}G -> ${a}G (+$((a-b))G)"
