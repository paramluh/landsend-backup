#!/bin/bash
# SPARE TEST of the Weka-served npl5 window (main OK 10-27 for slinky-13): on this spare, acting as slinky-13 with its REAL plans
#  0 refuse test: the canonical jobs symlinked to a dir OUTSIDE npl5-weka/<allowed> -> npl5_roll.py must exit 4 (no fetch, no delete)
#  1 canonical /scratch/amartyavarma/npl-append/npl5/jobs -> symlink -> /data/home/amartyavarma/ppm/pose5/npl5-weka/test-<spare>/jobs;
#    plans 1-3 (off-host rebuilds of slinky-13's, digests == dcp-17850) + its npl5 plan at the canonical plan dirs
#  2 npl5_prep.sh (PREP_AS=slinky-13): window fetch THROUGH the symlink (disk floor on /data), fresh_pose check rows 0-2 (raw parquet read through
#    the symlink), the trainer's micros (FreshPoseMultiDS / FreshRows, the DataLoader path) at the first npl5 row through the symlink
#  3 sim v2 (keep=1 + hold, relaunch from the held save, fake 429s) beside the daemon: 0 missing; drops land only inside the Weka test dir;
#    heartbeat snapshots every 20 s show the numbers moving INSIDE a pass (R2-9)
# Cleanup: exact paths only (the symlink is removed, never followed).
set -uo pipefail
W=/data/home/amartyavarma/ppm/pose5/npl5; B=$W/bin; S=/scratch/amartyavarma; ME=$(hostname -s)
WD=/data/home/amartyavarma/ppm/pose5/npl5-weka/test-$ME; T=$W/wekatest/run-${SLURM_JOB_ID:-x}; mkdir -p $T/kit $T/roll
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "== $(pt) $*" | tee -a $T/RESULT.txt; }
for d in ppm-fresh-pose ppm-fresh-pose-npl2 ppm-fresh-pose-npl3 ppm-fresh-pose-npl5 npl-append; do [ -e $S/$d ] && { say "ABORT: $S/$d exists on $ME"; exit 1; }; done
[ -e $WD ] && { say "ABORT: $WD exists"; exit 1; }
cleanup() {
  rm -rf "${S:?}/ppm-fresh-pose" "${S:?}/ppm-fresh-pose-npl2" "${S:?}/ppm-fresh-pose-npl3" "${S:?}/ppm-fresh-pose-npl5" "${S:?}/npl-append/npl5"
  rmdir "${S:?}/npl-append" 2>/dev/null; rm -rf "${WD:?}"; rm -rf "/data/home/amartyavarma/ppm/pose5/npl5-weka/test-wrong-${ME:?}"
  rmdir /data/home/amartyavarma/ppm/pose5/npl5-weka 2>/dev/null; true
}
export OMP_NUM_THREADS=1 NPL5_KIT_STATE=$T/kit NPL5_ROLL_DIR=$T/roll NPL5_WEKA_DIR=$WD
# ---- 1 canonical dirs
for d in ppm-fresh-pose ppm-fresh-pose-npl2 ppm-fresh-pose-npl3; do mkdir -p $S/$d; cp $W/rebuild13/slinky-13/$d/PLAN.json $W/rebuild13/slinky-13/$d/PLAN.npz $S/$d/; done
mkdir -p $S/ppm-fresh-pose-npl5; cp $W/plans/slinky-13/ppm-fresh-pose-npl5/PLAN.json $W/plans/slinky-13/ppm-fresh-pose-npl5/PLAN.npz $S/ppm-fresh-pose-npl5/
mkdir -p $S/npl-append/npl5
# ---- 0 refuse test (symlink to a dir outside the allowed one)
mkdir -p /data/home/amartyavarma/ppm/pose5/npl5-weka/test-wrong-$ME/jobs; ln -s /data/home/amartyavarma/ppm/pose5/npl5-weka/test-wrong-$ME/jobs $S/npl-append/npl5/jobs
$PY -I $B/npl5_roll.py window --end 19382 --upto-rel 10 --phys slinky-13 --key slinky-13 > $T/refuse.log 2>&1; rc=$?
say "0 refuse test: jobs -> npl5-weka/test-wrong-$ME (not the allowed $WD): npl5_roll.py rc $rc (4 expected); $(grep -o 'REFUSED.*' $T/refuse.log | cut -c1-160); files written there: $(find /data/home/amartyavarma/ppm/pose5/npl5-weka/test-wrong-$ME -type f | wc -l)"
rm -f "${S:?}/npl-append/npl5/jobs"
# ---- 1b the allowed symlink
mkdir -p $WD/jobs; ln -s $WD/jobs $S/npl-append/npl5/jobs
say "1 $S/npl-append/npl5/jobs -> $(readlink $S/npl-append/npl5/jobs)"
# ---- 2 prep through the symlink (SKIP_RESUME_MICRO: no npl3 parquet on this spare)
t0=$(date +%s); PREP_AS=slinky-13 SKIP_RESUME_MICRO=1 HF_HUB_OFFLINE=0 bash $B/npl5_prep.sh > $T/prep.log 2>&1; rc=$?
say "2 npl5_prep.sh as slinky-13 through the symlink: rc $rc in $(( $(date +%s) - t0 )) s; $(grep -h 'OK in\|FAIL' $T/prep.log | tail -1 | cut -c40-260)"
say "2 files on Weka $(find $WD/jobs -name '*.parquet' | wc -l), on the node-local canonical dir (excluding the symlink) $(find $S/npl-append/npl5 -mindepth 1 -maxdepth 1 ! -name jobs ! -name '.roll*' | wc -l); check: $(python3 -c "import json;d=json.load(open('$W/launch/prep/slinky-13/checks/CHECK-slinky-13.json'));print(len(d['rows']),'rows checked')" 2>&1)"
# ---- 3 sim + daemon (fake 429s) with heartbeat snapshots
mkdir -p $T/sim; ( while sleep 20; do [ -f $T/roll/slinky-13.json ] && python3 -c "import json,time;d=json.load(open('$T/roll/slinky-13.json'));print(time.strftime('%H:%M:%S'),d.get('phase'),d.get('heartbeat'),d.get('live_step'),d.get('covered_until'),d.get('ahead_steps'),d.get('fetched_in_pass'),d.get('r429'),d.get('alerts'))" >> $T/heartbeats.txt; done ) & HB=$!
$PY -I $B/sim_trainer.py --root $W/plans/slinky-13 --key slinky-13 --end 19382 --stop ${STOP:-18200} --audit13 --dir $T/sim --rate ${SIM_RATE:-0.5} --fast-rate ${SIM_FAST:-2} \
  --relaunch-at ${RELAUNCH_AT:-18150} --init-s 20 > $T/sim.out 2>&1 &
SIMP=$!
sleep 3
NPL5_FAKE_429=${FAKE429:-0.3} $PY -I $B/npl5_roll.py daemon --end 19382 --phys slinky-13 --key slinky-13 --ahead 60 --ckpt-dir $T/sim/ckpt --hold-dir $T/sim/hold \
  --pin-dir $T/sim/pin --log $T/sim/trainer.log --poll 5 --hf-rate ${HFRATE:-20} --threads ${THREADS:-8} > $T/daemon.out 2>&1 &
DP=$!
wait $SIMP; rc=$?
kill $DP 2>/dev/null; wait $DP 2>/dev/null; kill $HB 2>/dev/null
say "3 sim rc $rc: $(tail -n 1 $T/sim.out | cut -c1-300)"
say "3 daemon: passes $(grep -c 'ok:\|ROLL ' $T/daemon.out), 429s $(python3 -c "import json;print(json.load(open('$T/roll/slinky-13.json'))['r429'])"), alerts $(grep -o 'ROLL [A-Z+]*' $T/daemon.out | sort | uniq -c | tr '\n' ' '), dropped $(grep -o 'dropped [0-9]*' $T/daemon.out | awk '{s+=$2} END{print s}')"
say "3 Weka dir now $(du -sh $WD 2>/dev/null | cut -f1); files outside it touched: $(find $S/npl-append/npl5 -mindepth 1 ! -path "$S/npl-append/npl5/jobs" ! -name '.roll*' | wc -l)"
say "3 heartbeat snapshots (time phase heartbeat live covered ahead fetched_in_pass r429 alerts): $(wc -l < $T/heartbeats.txt) taken; see heartbeats.txt"
cleanup; say "cleanup done: $(ls -d $S/ppm-fresh-pose* $S/npl-append $WD 2>&1 | grep -vc 'No such')"
