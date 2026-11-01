#!/bin/bash
# aloe-v2pure-17352 (10-22): free node-local /scratch on the FT-(a) spare nodes (slinky-4 / 15 / 18 had 29-48 GB free; the FT needs ~340-430 GB
# per host). Deletes ONLY items verified superseded by an export or a redundant copy (checked 10-22 ~11:50 PT), via scratch_guard guard_rm;
# one line per deletion -> scratch-inspect/ft17352-cleanup-1022/DELETED.log. Run one task per node: srun -w <host> bash free_scratch.sh
source /data/home/amartyavarma/ppm/scratch_guard.sh
H=$(hostname -s); LOG=/data/home/amartyavarma/scratch-inspect/ft17352-cleanup-1022/DELETED.log; S=/scratch/amartyavarma
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M PT'; }
del() {  # del <path> <reason>
  local p=$1 why=$2 sz
  [ -e "$p" ] || { echo "$H absent $p"; return 0; }
  sz=$(du -s --block-size=1G "$p" | cut -f1)
  if guard_rm "$p"; then echo "$(pt) $H deleted $p (${sz}G) -- $why [aloe-v2pure-17352 FT-(a) spare-node space]" | tee -a $LOG
  else echo "$H guard REFUSED $p"; fi
}
W23051="resumable save of the FINISHED aloe-v2pure-ft FT job 23051 (rc 0, 200/200 steps, 10-17); its final step-200 weights = /data/home/amartyavarma/ppm/aloe-v2pure-ft/export-200 (COMPLETE, 83G)"
case $H in
  slinky-4)
    del $S/aloe-v2pure-ft/saves/step-00173 "$W23051; superseded by step 200"
    del $S/aloe-v2pure-ft/saves/step-00200 "$W23051";;
  slinky-15)
    del $S/aloe-v2pure-ft/saves/step-00173 "$W23051; superseded by step 200"
    del $S/aloe-v2pure-ft/saves/step-00200 "$W23051"
    for e in 33 67 100 133 167; do
      del $S/aloe-v2pure-ft/exports/export-$e "intermediate node-local weights export of FINISHED job 23051, superseded by its exported final step 200 (/data aloe-v2pure-ft/export-200); slinky-15's 8-of-24 shard set only"
    done;;
  slinky-18)
    del $S/fastinfer/ckpt/aloe-pose-ft "inference staging COPY of job 22091's export-430 (10-17); original node-local export-430 GCS-verified PASS on all 8 hosts (ppm/aloe-pose-ft/logs/gcs_verify)"
    W=/data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-ttaloe-from792-j11953/w-145
    if [ "$(du -sb $S/ckpt-stage/w-145 | cut -f1)" = "$(du -sb $W | cut -f1)" ] && [ "$(ls $S/ckpt-stage/w-145 | wc -l)" = "$(ls $W | wc -l)" ]; then
    del $S/ckpt-stage/w-145 "staging COPY (10-11) of /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-ttaloe-from792-j11953/w-145 (COMPLETE, 82 files, 92.2 GB on /data); bytes + file count equal"
    else echo "$H ckpt-stage/w-145 differs from the /data w-145: kept"; fi
    del $S/sam-aloe-ft-v2pure/ckpt/step-636 "resumable save of the FINISHED SAM 8B v2pure FT (step 636 = 1 epoch, 10-14); its step-636 weights = /data/home/amartyavarma/sam-aloe-ft-v2pure/export-step-636 + private HF InternalCan/sam-pose-aloe-v2pure-step636"
    del $S/qwen-s3-ckpt/q32-s3-alopw-from996-j20477/dcp-88 "node-local dcp-88 of q32-s3-alopw-from996-j20477 (10-14); its step-88 export = /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-alopw-from996-j20477/w-88 (COMPLETE, 32 distcp)";;
esac
echo "$H done; /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -d ' ')"
