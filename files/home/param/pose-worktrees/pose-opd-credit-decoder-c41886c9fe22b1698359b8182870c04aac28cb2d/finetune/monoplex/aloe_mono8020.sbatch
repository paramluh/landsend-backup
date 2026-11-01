#!/bin/bash
#SBATCH -J posemono8020 -p all -N8 -w slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-19 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-pose-mono-8020/logs/posemono8020-%j.log
# aloe-pose-mono-8020 (10-18): PPM 32B MONOPLEX pose FT (= job 23051's recipe, v2pure) on the Aloe handover pose rows (the qualifying tracks of
# job 22091's stores, one monoplex item per track) with the 10-18 prompt rule (voice clip p 0.8, persona text p 0.2, independent draws).
# 8 nodes x 8 H100 (world 64), the nodes of the 14B Voxtral FT (job 24193) after it ends (submitted with --dependency=afterany:24193).
# Gate: waits (<= 90 min) for READY (written once the code / env / plan checks pass) so the job holds its nodes from the moment they free.
# Stop: touch control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>). Resume: sbatch this file again (same nodes).
Q=/data/home/amartyavarma/ppm/aloe-pose-mono-8020
for i in $(seq 1 1080); do [ -f $Q/READY ] && break; [ -f $Q/ABORT ] && { echo "ABORT file present"; exit 1; }; sleep 5; done
[ -f $Q/READY ] || { echo "READY never written (90 min): exit"; exit 1; }
echo "== READY seen $(TZ=America/Los_Angeles date '+%F %T PT')"
exec bash $Q/aloe_mono8020_run.sh
