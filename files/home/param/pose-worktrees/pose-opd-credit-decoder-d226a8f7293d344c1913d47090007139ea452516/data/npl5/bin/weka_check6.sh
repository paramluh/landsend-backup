#!/bin/bash
# Dry check of a 2nd Weka-served host (slinky-6, approved 10-27) on a SPARE with slinky-6's REAL plans: (a) cross-host refuse: acting as slinky-6,
# the canonical jobs symlinked to slinky-13's Weka dir (target NOT created) -> npl5_roll.py must exit 4 before writing anything; (b) default
# allowed path for slinky-6 printed; (c) a small window fetch as slinky-6 through a symlink into a test dir (separate dir, lock, state);
# then fresh_pose check rows 0-2 through the symlink. Cleanup: exact paths only.
set -uo pipefail
W=/data/home/amartyavarma/ppm/pose5/npl5; B=$W/bin; S=/scratch/amartyavarma; ME=$(hostname -s)
WD=/data/home/amartyavarma/ppm/pose5/npl5-weka/test6-$ME; T=$W/wekatest/check6-${SLURM_JOB_ID:-x}; mkdir -p $T/kit $T/roll
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "== $(pt) $*" | tee -a $T/RESULT.txt; }
for d in ppm-fresh-pose ppm-fresh-pose-npl2 ppm-fresh-pose-npl3 ppm-fresh-pose-npl5 npl-append; do [ -e $S/$d ] && { say "ABORT: $S/$d exists on $ME"; exit 1; }; done
[ -e /data/home/amartyavarma/ppm/pose5/npl5-weka/slinky-13 ] && SKIP13=1 || SKIP13=0
export OMP_NUM_THREADS=1 NPL5_KIT_STATE=$T/kit NPL5_ROLL_DIR=$T/roll
for d in ppm-fresh-pose ppm-fresh-pose-npl2 ppm-fresh-pose-npl3; do mkdir -p $S/$d; cp $W/rebuild13/slinky-6/$d/PLAN.json $W/rebuild13/slinky-6/$d/PLAN.npz $S/$d/; done
mkdir -p $S/ppm-fresh-pose-npl5; cp $W/plans/slinky-6/ppm-fresh-pose-npl5/PLAN.json $W/plans/slinky-6/ppm-fresh-pose-npl5/PLAN.npz $S/ppm-fresh-pose-npl5/
mkdir -p $S/npl-append/npl5
# (a) cross-host refuse (target path only, never created)
ln -s /data/home/amartyavarma/ppm/pose5/npl5-weka/slinky-13/jobs $S/npl-append/npl5/jobs
$PY -I $B/npl5_roll.py window --end 19382 --upto-rel 3 --phys slinky-6 --key slinky-6 > $T/refuse.log 2>&1; rc=$?
say "a cross-host refuse: acting as slinky-6, jobs -> npl5-weka/slinky-13/jobs: rc $rc (4 expected): $(grep -o 'REFUSED.*' $T/refuse.log | cut -c1-200); slinky-13 dir created: $([ -e /data/home/amartyavarma/ppm/pose5/npl5-weka/slinky-13 ] && [ $SKIP13 = 0 ] && echo YES-BAD || echo no)"
rm -f "${S:?}/npl-append/npl5/jobs"
say "b default allowed dest for slinky-6: $($PY -c "print('/data/home/amartyavarma/ppm/pose5/npl5-weka/' + 'slinky-6' + '/jobs')") (WEKA_DIR = npl5-weka/<the host's own name>)"
# (c) small window as slinky-6 through a test symlink
mkdir -p $WD/jobs; ln -s $WD/jobs $S/npl-append/npl5/jobs
NPL5_WEKA_DIR=$WD HF_HUB_OFFLINE=0 $PY -I $B/npl5_roll.py window --end 19382 --upto-rel 3 --phys slinky-6 --key slinky-6 --hf-rate 30 > $T/window.log 2>&1; rc=$?
say "c window as slinky-6 through the symlink: rc $rc; $(grep -h 'ok:\|ROLL' $T/window.log | tail -1 | cut -c40-260)"
say "c files on Weka $(find $WD/jobs -name '*.parquet' | wc -l); node-local lock/state: $(ls -a $S/npl-append/npl5 | grep -c '^\.roll'); status dest_free_GB / scratch_free_GB: $(python3 -c "import json;d=json.load(open('$T/roll/slinky-6.json'));print(d.get('dest_free_GB'), d.get('scratch_free_GB'), d.get('dest_real'))")"
env FRESH_SRC=$S/npl-append/npl5/jobs FRESH_OUT=$S/ppm-fresh-pose-npl5 FRESH_SCAN=$T FRESH_TAG=npl5 \
  FRESH_PRIOR_SCANS=/data/home/amartyavarma/ppm/code-811/fresh/scan,/data/home/amartyavarma/ppm/code-811/fresh/append/npl2/scan,/data/home/amartyavarma/ppm/code-811/fresh/append/npl3/scan \
  FRESH_PRIOR_VIDEO=overlap FRESH_HOST=slinky-6 $PY -I /data/home/amartyavarma/ppm/pose5/code-811-s2p/fresh/fresh_pose.py check --rows 0,1,2 > $T/check.log 2>&1
say "c fresh_pose check rows 0-2 through the symlink: rc $? ($(grep -c 'check OK' $T/check.log) 'check OK')"
rm -rf "${S:?}/ppm-fresh-pose" "${S:?}/ppm-fresh-pose-npl2" "${S:?}/ppm-fresh-pose-npl3" "${S:?}/ppm-fresh-pose-npl5" "${S:?}/npl-append/npl5"
rmdir "${S:?}/npl-append" 2>/dev/null; rm -rf "${WD:?}"; rmdir /data/home/amartyavarma/ppm/pose5/npl5-weka 2>/dev/null
say "cleanup: leftovers $(ls -d $S/ppm-fresh-pose* $S/npl-append $WD 2>/dev/null | wc -l)"
