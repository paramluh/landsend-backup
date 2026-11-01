#!/bin/bash
# MONOFT dry run (10-22, main run 27653 TRAINING): the init distribution end to end on the CURRENT latest save, as an overlap step of the main
# job (CPU only: -c 2, --cpu-bind=none, nice 19 / ionice idle conversion, fetch capped at 2 x 100 MB/s per host). Stage + init are node-local
# and deleted at the end (MF_DRY=1); dcp-<S> is only read. usage: run_dry.sh <S> <8 FT hosts>
S=$1; FT=$2; D=/data/home/amartyavarma/ppm/monoft/dryrun/$S; mkdir -p $D
rm -rf $D/ctl $D/tools; rsync -a --exclude __pycache__ /data/home/amartyavarma/ppm/monoft/tools/ $D/tools/   # frozen copy (never run a live-edited script)
export MF_S=$S MF_FT_HOSTS=$FT MF_ALL_HOSTS=$(cat /data/home/amartyavarma/ppm/code-811/nodes16_s7.txt) MF_CTL=$D/ctl \
  MF_INIT=/scratch/amartyavarma/monoft-dry-$S-init MF_STAGE=/scratch/amartyavarma/monoft-dry-$S-stage MF_PORT=29892 MF_MBPS=${MF_MBPS:-100} MF_WORKERS=${MF_WORKERS:-3} \
  MF_DRY=1 MF_FINAL_SAVE=0 MF_TOOLS=$D/tools
echo "== dry run S=$S FT=$FT start $(TZ=America/Los_Angeles date '+%F %T PT')"
srun --jobid 27653 --overlap -N16 -n16 --ntasks-per-node=1 --cpu-bind=none -c2 --kill-on-bad-exit=0 \
  bash $D/tools/init_dist.sh
rc=$?; echo "== dry run rc $rc $(TZ=America/Los_Angeles date '+%F %T PT')"; exit $rc
