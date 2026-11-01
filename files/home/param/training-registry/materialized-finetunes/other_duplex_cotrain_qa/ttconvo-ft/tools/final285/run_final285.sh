#!/bin/bash
# ttconvo-ft final state (10-21): dcp-285 -> node-local export-285 (CPU, exact) -> private HF InternalCan/ppm-ttconvo-ft/export-285/ (verify).
# Everything runs as CPU-only overlap steps of the job holding the 16 hosts (JOB, the PPM 14B run): -c 4 / -c 1, --gres=none, --cpu-bind=none,
# nice 19, ionice idle. Launcher only on the login node. Deletes nothing.
#   JOB=<job on the 16 hosts> bash run_final285.sh [convert] [upload]
set -uo pipefail
T=/data/home/amartyavarma/ppm/ttconvo-ft/tools/final285; B=/data/home/amartyavarma/ppm/ttconvo-ft
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
HOSTS=slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-17,slinky-19
: "${JOB:?JOB not set}"
NICE="nice -n 19 ionice -c3"
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
ST=$(squeue -h -j $JOB -o "%T %N")
[ "$(echo $ST | cut -d' ' -f1)" = RUNNING ] || { echo "[$(pt)] job $JOB not running ($ST)"; exit 1; }
WHAT=${*:-convert upload}
if [[ " $WHAT " == *" convert "* ]]; then
  echo "[$(pt)] convert: dcp-285 -> export-285 on 16 hosts (overlap steps of job $JOB)"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 srun --jobid $JOB --overlap -N16 -n16 -w $HOSTS --ntasks-per-node=1 -c 2 --mem=16G --gres=none --cpu-bind=none \
      -J ttc285-export --kill-on-bad-exit=0 -t 60 $NICE $PY $T/export_from_dcp.py || { echo "[$(pt)] convert FAILED"; exit 1; }
  srun --jobid $JOB --overlap -N1 -n1 -w slinky-0 -c 2 --mem=2G --gres=none --cpu-bind=none -J ttc285-export -t 5 $NICE \
      $PY $T/export_from_dcp.py --finalize || { echo "[$(pt)] finalize FAILED"; exit 1; }
  echo "[$(pt)] convert done"
fi
if [[ " $WHAT " == *" upload "* ]]; then
  # hf_local_exports.sh overlap mode = the run's own uploader (2 workers/host, gentle xet); its throttle control files are only written by
  # the watcher during ITS launches (none pending: plan 47,94,195 done, 296 never written), so set them to the fixed 2 workers here.
  for h in $(echo $HOSTS | tr , ' '); do echo 2 > $B/hf_backup/control/$h.workers; done
  export HF_XET_CHUNK_CACHE_SIZE_BYTES=0                    # no download chunk cache growth on /scratch (verify streams 3 files back)
  TTC_LAUNCH=overlap TTC_JOB=$JOB TTC_EXPORTS=285 TTC_LAUNCH_ID=final285-overlap-$JOB-$(date +%s) bash $B/hf_local_exports.sh
  rc=$?
  for h in $(echo $HOSTS | tr , ' '); do echo 0 > $B/hf_backup/control/$h.workers; done
  echo "[$(pt)] upload rc $rc"
  exit $rc
fi
