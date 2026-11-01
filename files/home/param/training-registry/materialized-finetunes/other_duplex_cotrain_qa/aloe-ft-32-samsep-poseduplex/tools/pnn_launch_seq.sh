#!/bin/bash
# PNN launch sequence (login node: srun df / sbatch only). usage: bash pnn_launch_seq.sh <tag w-N>
set -uo pipefail
T=$1; P=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex; HOSTS=slinky-1,slinky-2,slinky-6,slinky-15,slinky-7,slinky-13,slinky-14,slinky-17; NEED=149
grep -q '^STEPS=498' /data/home/amartyavarma/ppm/control/POSEDUPLEX_GO || { echo "ABORT: no GO with STEPS=498"; exit 1; }
squeue -h -j 29863 | grep -q . && { echo "ABORT: v5 29863 still running"; exit 1; }
timeout 300 srun -p all -w $HOSTS -N8 -n8 --ntasks-per-node=1 -c1 --mem=200M -t 5 bash -c 'echo "$(hostname -s) $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9) $([ -d /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850 ] && echo has17850 || echo no17850)"' 2>/dev/null | sort > $P/logs/pnn_room_before.txt
cat $P/logs/pnn_room_before.txt
SHORT=$(awk -v n=$NEED '$2 < n {print $1}' $P/logs/pnn_room_before.txt | paste -sd,)
if [ -n "$SHORT" ]; then
  echo "short hosts: $SHORT -> room_clean.sh there (D1 dcp-17850, then the user's clear-list)"
  NS=$(echo $SHORT | tr , '\n' | wc -l)
  timeout 1200 srun -p all -w $SHORT -N$NS -n$NS --ntasks-per-node=1 -c2 --mem=500M -t 20 --export=ALL,NEED=$NEED bash $P/tools/room_clean.sh 2>&1 | grep -v "^srun" | cut -c1-220
  timeout 300 srun -p all -w $HOSTS -N8 -n8 --ntasks-per-node=1 -c1 --mem=200M -t 5 bash -c 'echo "$(hostname -s) $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)"' 2>/dev/null | sort > $P/logs/pnn_room_after.txt
  cat $P/logs/pnn_room_after.txt
  STILL=$(awk -v n=$NEED '$2 < n {printf "%s (%d GB short) ", $1, n - $2}' $P/logs/pnn_room_after.txt)
  [ -n "$STILL" ] && { echo "STILL SHORT: $STILL -> not launching (preflight would abort)"; exit 2; }
fi
J1=$(POSEDUP_HOSTS=$HOSTS sbatch --parsable -N8 -w $HOSTS --export=ALL $P/stage_init_v5.sbatch $T); echo "stage job $J1"
POSEDUP_INIT_TAG=$T QQA_STEPS_SET=498 bash $P/launch_poseduplex.sh slinky-1 slinky-2 slinky-6 slinky-15 slinky-7 slinky-13 slinky-14 slinky-17 afterok:$J1
