#!/bin/bash
# aloe-v2vox-asr-w15973-textB launcher (login node: sbatch only). usage: bash launch.sh
set -euo pipefail
N=/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973-textB
INIT=/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973
AD=/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973/adapter-init/asrppm-w15973-w-1441
[ -f $INIT/COMPLETE ] && [ -f $AD/COMPLETE ] || { echo "ABORT: init / adapter not COMPLETE"; exit 1; }
grep -qx "VOX_ADAPTER=$AD" $N/aloe_textB.env && grep -qx "ALOE_B_TEXT_ONLY=1" $N/aloe_textB.env || { echo "ABORT: env"; exit 1; }
HOSTS=slinky-7,slinky-15,slinky-18
J=$(QQA_INIT=$INIT sbatch --parsable -N3 -w $HOSTS --cpus-per-task=5 --mem=540G $N/aloe_textB.sbatch)
B=$(sbatch --parsable --dependency=afterok:$J -N3 -w $HOSTS $N/hf_local_exports.sbatch)
printf '{"job": "%s", "nodes": "%s", "cpt": 5, "mem": "540G", "init": "%s", "adapter": "%s (= job 26232 adapter-init copy: ASR asrppm-w15973 job 26064 step 1441, sha 2b7c543cad150d96)", "switch": "ALOE_B_TEXT_ONLY=1", "backup_job": "%s (afterok, HF private InternalCan/ppm-aloe-v2vox-asr-w15973-textB; no node-local deletions)", "time": "%s"}\n' \
  "$J" "$HOSTS" "$INIT" "$AD" "$B" "$(TZ=America/Los_Angeles date '+%F %T PT')" > $N/LAUNCHED_JOB
cat $N/LAUNCHED_JOB
