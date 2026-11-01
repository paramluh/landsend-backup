#!/bin/bash
# aloe-v2vox-asr-w15973 launcher (login node: copy 126 MB adapter + sbatch only). usage: bash launch.sh <ASR export step> <ASR logged sha16> [dependency]
set -euo pipefail
N=/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973; S=$1; SHA=$2; DEP=${3:-}
SRC=/data/home/amartyavarma/ppm/asr-adapter/ckpt-weka/asrppm-w15973/w-$S; DST=$N/adapter-init/asrppm-w15973-w-$S
[ -f $SRC/COMPLETE ] && [ -f $SRC/HF_BACKUP.json ] && grep -q '"PASS": true' $SRC/HF_BACKUP.json || { echo "ABORT: $SRC not COMPLETE + HF-verified"; exit 1; }
if [ ! -f $DST/COMPLETE ]; then
  rm -rf $DST.tmp; cp -a $SRC $DST.tmp
  for f in .metadata COMPLETE __0_0.distcp train_state.json HF_BACKUP.json; do cmp $SRC/$f $DST.tmp/$f; done
  mv $DST.tmp $DST
fi
sed -i "s#^VOX_ADAPTER=.*#VOX_ADAPTER=$DST#; s#^VOX_ADAPTER_SHA=.*#VOX_ADAPTER_SHA=$SHA#" $N/aloe_v2vox_raw.env
grep -E "^VOX_ADAPTER(_SHA)?=" $N/aloe_v2vox_raw.env
HOSTS=slinky-5,slinky-6,slinky-8,slinky-10,slinky-11,slinky-16,slinky-17,slinky-19
J=$(QQA_INIT=/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973 sbatch --parsable ${DEP:+--dependency=$DEP} -N8 -w $HOSTS --cpus-per-task=5 --mem=540G $N/aloe_v2vox_raw.sbatch)
B=$(sbatch --parsable --dependency=afterok:$J -N8 -w $HOSTS $N/hf_local_exports.sbatch)
printf '{"job": "%s", "nodes": "%s", "cpt": 5, "mem": "540G", "init": "/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973", "adapter": "%s (ASR asrppm-w15973 job 26064 step %s, sha %s; copied from %s)", "dependency": "%s", "backup_job": "%s (afterok, HF private InternalCan/ppm-aloe-v2vox-asr-w15973)", "time": "%s"}\n' \
  "$J" "$HOSTS" "$DST" "$S" "$SHA" "$SRC" "$DEP" "$B" "$(TZ=America/Los_Angeles date '+%F %T PT')" > $N/LAUNCHED_JOB
cat $N/LAUNCHED_JOB
