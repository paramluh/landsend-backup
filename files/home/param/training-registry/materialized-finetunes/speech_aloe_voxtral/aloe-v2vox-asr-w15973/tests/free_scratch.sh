#!/bin/bash
# aloe-v2vox-asr-w15973 (10-20 ~18:55 PT): free node /scratch for the 8-node FT's node-local exports (need >= 100 GB/node; 26221 aborted at 52 GB).
# Deletes ONLY stale Qwen s3/s4 training ckpts whose step (or a later superseding step) is exported on /data (COMPLETE + .metadata; verified
# before this ran), through scratch_guard guard_rm; one line per deletion -> DELETED-1019-hfb-hold.log.
source /data/home/amartyavarma/ppm/scratch_guard.sh
H=$(hostname -s); LOG=/data/home/amartyavarma/scratch-inspect/DELETED-1019-hfb-hold.log
Q3=/scratch/amartyavarma/qwen-s3-ckpt; Q4=/scratch/amartyavarma/qwen-s4-ckpt; E3=/data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka; E4=/data/home/amartyavarma/qwen-1ch/stage4/ckpt-weka
declare -A WHY
WHY[$Q3/q32-s3-ttpw-from792-j20336/dcp-996]="exported: /data w-996 (same step) + HF InternalCan/tt-s3-ttpw-j20336 w-996 (records.jsonl PASS); evidence $E3/q32-s3-ttpw-from792-j20336/w-996 COMPLETE+.metadata"
WHY[$Q3/q32-s3-ttpw-from792-j20336/dcp-960]="superseded by later step 996 of the same run, exported: /data w-996 + HF InternalCan/tt-s3-ttpw-j20336 w-996; evidence $E3/q32-s3-ttpw-from792-j20336/w-996 COMPLETE+.metadata"
WHY[$Q4/q32-s4-val-s3x5900-j9411/dcp-6339]="exported: /data w-6339 (same step); evidence $E4/q32-s4-val-s3x5900-j9411/w-6339 COMPLETE+.metadata"
WHY[$Q3/q32-s3-alopw8-from996-j20490/dcp-129]="exported: /data w-129 (same step); evidence $E3/q32-s3-alopw8-from996-j20490/w-129 COMPLETE+.metadata"
case $H in
  slinky-5) T="$Q3/q32-s3-alopw8-from996-j20490/dcp-129";;
  slinky-6|slinky-8|slinky-16|slinky-17) T="$Q3/q32-s3-ttpw-from792-j20336/dcp-996 $Q3/q32-s3-ttpw-from792-j20336/dcp-960";;
  slinky-10) T="$Q3/q32-s3-ttpw-from792-j20336/dcp-996 $Q4/q32-s4-val-s3x5900-j9411/dcp-6339";;
  slinky-11|slinky-19) T="$Q3/q32-s3-ttpw-from792-j20336/dcp-996";;
  *) echo "$H: not a target host"; exit 1;;
esac
for d in $T; do
  [ -d "$d" ] || { echo "$H skip $d (absent)"; continue; }
  sz=$(du -s --block-size=1G "$d" | cut -f1)
  if guard_rm "$d"; then
    echo "$(TZ=America/Los_Angeles date '+%m-%d %H:%M PT') $H deleted $d (${sz}G; ${WHY[$d]}) for the 8-node aloe-v2vox-asr-w15973 FT node-local exports; free now $(df -BG --output=avail /scratch | tail -1 | tr -d ' ')" | tee -a $LOG
  else echo "$H guard refused $d"; fi
done
echo "$H /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -d ' ')"
