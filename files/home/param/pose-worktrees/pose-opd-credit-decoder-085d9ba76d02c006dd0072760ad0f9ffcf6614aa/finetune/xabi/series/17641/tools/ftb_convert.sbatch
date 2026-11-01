#!/bin/bash
#SBATCH -J ftb-convert -p all -N16 --ntasks-per-node=1 -c 4 --mem=24G --gres=none -t 2:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-17641/logs/ftb_convert-%j.log
# aloe-v2pure-17641 FT-(b) init (10-22): the 8,1,1 append run's FINAL save dcp-$FTB_STEP (node-local on the 16 PPM hosts) ->
#  (1) PROTECT: hard links /scratch/amartyavarma/ppm/ckpt-keep-ft/dcp-$FTB_STEP (= the coordinator's dcp-17352 keep; no extra space while the original
#      exists; the run dir itself is only read);
#  (2) the local inference/init format on /data (ckpt_local_export.py, slot-aware copy: the 8,1,1 run records slinky-7 under slinky-10's logical
#      slot) -> /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-$FTB_STEP, COMPLETE written by --finalize;
#  (3) prints each host's /scratch free at its start (SCRATCHFREE0: ftb_chain.sh picks the FT hosts at once) and end (SCRATCHFREE). CPU-only, nice 19. Submitted by ftb_chain.sh.
set -uo pipefail
: "${FTB_STEP:?}"
export FTB_STEP PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python X=/data/home/amartyavarma/ppm/aloe-v2pure-17641/tools/ckpt_local_export_slot.py
export OUT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-$FTB_STEP
echo "== ftb_convert job $SLURM_JOB_ID $(TZ=America/Los_Angeles date '+%F %T PT') dcp-$FTB_STEP -> $OUT; /data $(df -BG --output=avail /data | tail -1 | tr -d ' ') free"
srun -N16 -n16 --ntasks-per-node=1 -c 4 --mem=24G --cpu-bind=none --kill-on-bad-exit=0 bash -c '
H=$(hostname -s); SRC=/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-$FTB_STEP; K=/scratch/amartyavarma/ppm/ckpt-keep-ft/dcp-$FTB_STEP
echo "$H SCRATCHFREE0 $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)"   # at start: ftb_chain.sh picks the FT hosts from these
[ -f $SRC/COMPLETE ] || [ -f $K/COMPLETE ] || { echo "$H ABORT: dcp-$FTB_STEP not COMPLETE here"; exit 1; }
if [ -f $SRC/COMPLETE ] && [ ! -f $K/COMPLETE ]; then
  mkdir -p $K.tmp && ln -f $SRC/* $K.tmp/ && mv $K.tmp $K && echo "$H protected: $(ls $K | wc -l) files hard-linked -> $K"
fi
[ -f $K/COMPLETE ] || { echo "$H ABORT: keep copy incomplete"; exit 1; }
AS=""; $PY -c "import json,sys; sys.exit(0 if sys.argv[1] in json.load(open(sys.argv[2])).get(\"hosts\", []) else 1)" $H $K/train_state.json || { [ $H = slinky-7 ] && AS=slinky-10; }   # hosts list only (a bare grep matched slinky-7 in node_swap/digest maps)
CKX_HOST_AS=$AS nice -n 19 $PY $X --src $K --out $OUT || exit 1
echo "$H SCRATCHFREE $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)"'
rc=$?
echo "== convert rc $rc $(TZ=America/Los_Angeles date '+%T PT')"
# a failed conversion leaves no partial copy on /data (our own just-written files: e.g. rank 0's host refused by the /data floor while others wrote)
[ $rc = 0 ] || { [ -f $OUT/COMPLETE ] || { rm -rf --one-file-system $OUT; echo "== removed the partial $OUT; /data $(df -BG --output=avail /data | tail -1 | tr -d ' ') free"; }; exit $rc; }
srun -N1 -n1 -c 1 --mem=4G --cpu-bind=none nice -n 19 $PY $X --out $OUT --finalize || exit 1
echo "== COMPLETE $(cat $OUT/COMPLETE) $(du -sBG $OUT | cut -f1); /data $(df -BG --output=avail /data | tail -1 | tr -d ' ') free $(TZ=America/Los_Angeles date '+%T PT')"
