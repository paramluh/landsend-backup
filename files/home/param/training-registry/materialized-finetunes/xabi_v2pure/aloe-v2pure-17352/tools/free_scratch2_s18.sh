#!/bin/bash
# aloe-v2pure-17352 (10-22) slinky-18 second pass (= free_scratch.sh's rules): ckpt-stage/w-145 was KEPT (differs from the /data w-145).
source /data/home/amartyavarma/ppm/scratch_guard.sh
H=$(hostname -s); LOG=/data/home/amartyavarma/scratch-inspect/ft17352-cleanup-1022/DELETED.log; S=/scratch/amartyavarma
[ "$H" = slinky-18 ] || { echo "ABORT: slinky-18 only"; exit 1; }
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M PT'; }
del() { local p=$1 why=$2 sz; [ -e "$p" ] || { echo "$H absent $p"; return 0; }; sz=$(du -s --block-size=1G "$p" | cut -f1)
  if guard_rm "$p"; then echo "$(pt) $H deleted $p (${sz}G) -- $why [aloe-v2pure-17352 FT-(a) spare-node space]" | tee -a $LOG; else echo "$H guard REFUSED $p"; fi; }
[ -f /data/home/amartyavarma/qwen-1ch/stage2/ckpt-weka/q32-s2-1ch-plainpt-noadapt-hiLR-lrt3e-5-j10569/w-2348/COMPLETE ] && \
  del $S/qwen-s2-ckpt/q32-s2-1ch-plainpt-noadapt-hiLR-lrt3e-5-j10569/dcp-2348 "node-local dcp-2348 of q32-s2-1ch-plainpt-noadapt-hiLR-lrt3e-5-j10569 (10-09); its step-2348 export = /data/home/amartyavarma/qwen-1ch/stage2/ckpt-weka/q32-s2-1ch-plainpt-noadapt-hiLR-lrt3e-5-j10569/w-2348 (COMPLETE)"
del $S/sam-aloe-ft-posev2/init-step-61258-fp32 "init CACHE (10-13) of SAM stage-2 step 61258 fp32 (sam-aloe-ft-posev2/tools/init.sbatch: hard link / HF dl_init.py); the finished posev2 FT's input only; same file also on slinky-4"
echo "$H done; /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -d ' ')"
