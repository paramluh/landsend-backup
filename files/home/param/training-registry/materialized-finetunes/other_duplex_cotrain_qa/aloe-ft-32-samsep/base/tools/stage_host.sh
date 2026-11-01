#!/bin/bash
# ALOEFT32 base: one task per host (slinky-18, slinky-4): stage dcp-$STEP from HF -> node-local bf16 shard copy, then finalize (COMPLETE).
# env: STEP (default 16861), REPO (default InternalCan/ppm-v1-b)
set -uo pipefail
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
B=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/base
STEP=${STEP:-16861}; REPO=${REPO:-InternalCan/ppm-v1-b}
INIT=/scratch/amartyavarma/aloe-samsep-$STEP-init
DL=/scratch/amartyavarma/aloe-samsep-$STEP-dl
H=$(hostname -s)
case $H in slinky-18) W=24;; slinky-4) W=10;; *) W=8;; esac   # coordinator 10-24: slinky-18 24 streams, slinky-4 ~10 (/scratch >= 30 GB)
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$DL/xet-cache HF_XET_CHUNK_CACHE_SIZE_BYTES=0 PYTHONUNBUFFERED=1
echo "[$(TZ=America/Los_Angeles date '+%m-%d %T PT')] $H start step $STEP ($REPO) W=$W; /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9) GB"
source /data/home/amartyavarma/ppm/scratch_guard.sh
nice -n 10 ionice -c2 -n7 $PY $B/tools/stage_17850.py --step $STEP --repo $REPO --init $INIT --dl $DL/raw --ctl $B/ctl --workers $W > $B/logs/stage-$STEP-$H.log 2>&1 || { echo "$H STAGE FAILED: $(tail -n 3 $B/logs/stage-$STEP-$H.log)"; exit 1; }
tail -n 1 $B/logs/stage-$STEP-$H.log
[ -d $DL/xet-cache ] && guard_rm $DL/xet-cache
$PY $B/tools/verify_17850.py --step $STEP --init $INIT --ctl $B/ctl > $B/logs/finalize-$STEP-$H.log 2>&1 || { echo "$H FINALIZE FAILED: $(tail -n 3 $B/logs/finalize-$STEP-$H.log)"; exit 1; }
tail -n 1 $B/logs/finalize-$STEP-$H.log
echo "[$(TZ=America/Los_Angeles date '+%m-%d %T PT')] $H done; init $(du -sBG $INIT | cut -f1); /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9) GB"
